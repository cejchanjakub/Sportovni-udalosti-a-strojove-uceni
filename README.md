# dp_sazeni_ml

> ML pipeline pro predikci výsledků zápasů anglické Premier League a identifikaci value betů.

Diplomová práce – Technická univerzita v Liberci | Python 3.11 | Sezóny 2015/16–2025/26

---

## Obsah

- [Přehled](#přehled)
- [Architektura](#architektura)
- [Modely](#modely)
- [Feature engineering](#feature-engineering)
- [Rychlý start](#rychlý-start)
- [Struktura projektu](#struktura-projektu)
- [Data](#data)
- [Přetrénování modelů](#přetrénování-modelů)
- [API dokumentace](#api-dokumentace)
- [Technické poznámky](#technické-poznámky)

---

## Přehled

Projekt implementuje kompletní ML pipeline pro predikci výsledků zápasů EPL a odvozování férových sázkových kurzů. Výstupem je funkční inference systém s webovým rozhraním, který umožňuje:

- Predikci výsledku zápasu (1X2) s kalibrovanými pravděpodobnostmi
- Generování O/U kurzů pro 15 sázkových trhů (góly, rohy, fauly, karty, střely)
- Identifikaci potenciálních value betů porovnáním modelových a bookmakerských kurzů
- Zohlednění statistik konkrétního rozhodčího u predikce faulů a karet

---

## Architektura

Pipeline je rozdělena do čtyř kroků:

```
1. download_raw.py           → stáhne CSV z football-data.co.uk
2. merge_raw_seasons.py
   build_processed_all.py    → předzpracování a normalizace
3. add_coaches_to_matches.py → doplnění trenérských dat
4. build_features_all.py     → feature engineering (ELO, forma, H2H, ...)
```

Po přepočtu featur se trénují modely (`train_all.bat`) a spouští inference server (`uvicorn`).

---

## Modely

### 1X2 klasifikační model

| Parametr | Hodnota |
|---|---|
| Algoritmus | CalibratedClassifierCV (LogReg + GradientBoosting) |
| Kalibrace | Isotonic regression |
| Počet featur | 53 |
| Artefakty | `artifacts/v1_model_freeze/1x2/` |

### 15 count modelů (O/U trhy)

| Trh | Target | Distribuce | Alpha (NB2) |
|---|---|---|---|
| Góly celkem | total_goals | Poisson | — |
| Góly domácí | FTHG | Poisson | — |
| Góly hosté | FTAG | Poisson | — |
| Rohy celkem | total_corners | NegBin | 0.0084 |
| Rohy domácí | HC | NegBin | 0.0639 |
| Rohy hosté | AC | NegBin | 0.0855 |
| Fauly celkem | total_fouls | NegBin | 0.0046 |
| Fauly domácí | HF | NegBin | 0.0100 |
| Fauly hosté | AF | NegBin | 0.0026 |
| Žluté karty celkem | total_cards | NegBin | 0.0116 |
| Žluté karty domácí | HY | NegBin | 0.0100 |
| Žluté karty hosté | AY | NegBin | 0.0100 |
| Střely na branku celkem | total_shots_on_target | Poisson | — |
| Střely na branku domácí | HST | Poisson | — |
| Střely na branku hosté | AST | Poisson | — |

> Alpha parametr NB2 je odhadován metodou momentů (MoM) z Pearsonových residuálů Poisson fitu na trénovací množině.

---

## Feature engineering

Featury jsou počítány v `src/build_features_all.py`:

| Skupina | Featury | Poznámka |
|---|---|---|
| **ELO** | `elo_home`, `elo_away`, `elo_diff` | base=1500, K=20, home_adv=50 |
| **Rolling forma** | `home/away_{prefix}_for/against_roll{3,5,10}` | Okna 3, 5, 10 zápasů |
| **Body** | `home/away_points_roll{3,5,10}` | Průměrný počet bodů |
| **Tabulka** | `home/away_table_pos`, `table_pos_diff` | Stav před zápasem |
| **Odpočinek** | `home/away_days_rest`, `is_midweek` | Dny od posledního zápasu |
| **Trenér** | `HomeCoachTenureDays`, `NewHomeCoach_30`, ... | Ze souboru managers CSV |
| **Rozhodčí** | `ref_fouls_avg_last20`, `ref_cards_avg_last20` | Posl. 20 zápasů; jen fouls+cards |
| **H2H** | `h2h_avg_yellow_last{3,5}`, `h2h_avg_goals_last{3,5}`, ... | Vzájemné zápasy, oba směry |

---

## Rychlý start

### Instalace závislostí

```bash
pip install -r requirements.txt
```

### Kompletní přepočet pipeline

```powershell
# 1. Stažení dat
python src/download_raw.py

# 2. Předzpracování
python src/merge_raw_seasons.py
python src/build_processed_all.py

# 3. Trenérské featury
python src/add_coaches_to_matches.py

# 4. Feature engineering (trvá 5–10 min kvůli H2H výpočtu)
python src/build_features_all.py

# 5. Trénování modelů
.\train_all.bat

# 6. Live features (budoucí zápasy)
python -m src.inference.refresh_live_features_from_api --days-ahead 14

# 7. Spuštění API
python -m uvicorn src.api_main:app --reload
```

Webové UI je dostupné na **http://localhost:8000/ui**

> ⚠️ Po každém spuštění `build_features_all.py` je nutné znovu spustit `refresh_live_features_from_api` –
> `build_features_all.py` přepíše `live_features.csv` historickými daty.

---

## Struktura projektu

```
dp_sazeni_ml/
├── src/
│   ├── download_raw.py
│   ├── merge_raw_seasons.py
│   ├── build_processed_all.py
│   ├── add_coaches_to_matches.py
│   ├── build_features_all.py              # Feature engineering (ELO, H2H, rozhodčí, trenér)
│   ├── model_count_glm.py                 # Trénování GLM count modelů
│   ├── model_1X2.py                       # Trénování 1X2 modelu
│   ├── api_main.py                        # FastAPI aplikace
│   ├── line_generator.py                  # Generování O/U linek a kurzů
│   ├── ui/
│   │   └── index.html                     # Webové UI
│   └── inference/
│       ├── registry.py                    # Registrace všech 16 trhů
│       ├── model_loader.py                # Načítání modelů + alpha z meta.json
│       ├── refresh_live_features_from_api.py
│       ├── Team_mapper.py                 # Mapování názvů týmů API → historická data
│       ├── predict_1x2_from_live_features.py
│       ├── providers/
│       │   └── football_data_provider.py
│       └── services/                      # 15 service tříd (goals, corners, fouls, ...)
├── artifacts/
│   └── v1_model_freeze/
│       ├── 1x2/                           # model.joblib, scaler.joblib, features.json, meta.json
│       ├── goals__total_goals/            # model.sm, features.json, meta.json
│       └── ...
├── data/
│   ├── raw/                               # Surová CSV – negitováno, generuje download_raw.py
│   ├── processed/
│   │   └── epl_main_managers_2015_2026.csv  # ⭐ Ručně sestavená data trenérů (gitováno)
│   └── features/                          # Výstupní featury – negitováno
├── docs/
│   ├── data_dictionary.md
│   └── data_sources.md
├── requirements.txt
├── train_all.bat
└── README.md
```

---

## Data

### Zdrojová data

| Zdroj | Obsah | Umístění |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | Výsledky, statistiky, kurzy EPL | `data/raw/EPL_*.csv` |
| Wikipedia (ručně sestaveno) | Trenéři EPL 2015–2026 | `data/processed/epl_main_managers_2015_2026.csv` |

### Rozdělení dat

| Split | Sezóny | Zápasy |
|---|---|---|
| Train | 2015/16 – 2021/22 | ~2 660 |
| Val | 2022/23 | ~380 |
| Test | 2023/24 | ~380 |
| Live | Budoucí fixtures | ~20 |

---

## Přetrénování modelů

### Všechny modely najednou

```powershell
python src/build_features_all.py
.\train_all.bat
```

### Jednotlivé modely

```powershell
# Poisson modely (góly, střely)
python src/model_count_glm.py --target total_goals --prefix goals --family poisson --save_artifacts

# NegBin modely (rohy, fauly, karty) – alpha se odhaduje automaticky
python src/model_count_glm.py --target total_corners --prefix corners --family negbin --save_artifacts
python src/model_count_glm.py --target total_fouls   --prefix fouls   --family negbin --save_artifacts
python src/model_count_glm.py --target total_cards   --prefix yellow  --family negbin --save_artifacts

# 1X2 model
python src/model_1X2.py --save_artifacts
```

---

## API dokumentace

Po spuštění je Swagger UI dostupné na **http://localhost:8000/docs**

### Endpointy

| Endpoint | Metoda | Popis |
|---|---|---|
| `GET /ui` | GET | Webové rozhraní |
| `GET /fixtures` | GET | Seznam nadcházejících zápasů |
| `POST /predict` | POST | Predikce kurzů pro zápas |
| `GET /referees` | GET | Seznam rozhodčích se statistikami |
| `GET /health` | GET | Health check |

### Příklad predikce

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "match": {
      "utc_date": "2026-04-19T14:00:00Z",
      "home_team": "Arsenal FC",
      "away_team": "Manchester City FC",
      "referee": "M Oliver"
    },
    "markets": ["1x2", "goals_total", "fouls_total", "cards_total"]
  }'
```

### Dostupné trhy

```
1x2
goals_total   goals_home   goals_away
corners_total corners_home corners_away
fouls_total   fouls_home   fouls_away
cards_total   cards_home   cards_away
shots_total   shots_home   shots_away
```

> Parametr `referee` ovlivňuje pouze trhy `fouls_*` a `cards_*`. Pro ostatní trhy je ignorován.

---

## Technické poznámky

**NB2 alpha** je odhadována metodou momentů z Pearsonových residuálů Poisson fitu.
MLE solver (newton) diverguje pro nízkou overdisperzi – proto je použita MoM metoda.

**Team mapper** (`Team_mapper.py`) mapuje názvy týmů z API (`Arsenal FC`) na historické názvy (`Arsenal`).
Bez tohoto mapování ELO lookup nenajde tým a vrátí defaultní hodnotu 1500.

**Model loader** vrací trojici `(model, features, alpha)`.
Alpha se načítá z `meta.json` a předává do `line_generator` pro správný výpočet NB2 pravděpodobností.

**Count GLM modely** nevyžadují standardizaci – `sm.add_constant(has_constant="add")` se přidává vždy.

**1X2 model** používá sklearn s `StandardScaler` fitovaným pouze na trénovací množině.

**Live features** mají 189 sloupců (shodně s train/val/test po přidání H2H featur).

---

## Závislosti

```
fastapi
uvicorn
statsmodels
scikit-learn
pandas
numpy
joblib
requests
pydantic
```

Viz `requirements.txt` pro přesné verze.

---

## Autor

Jakub Čejchan – Mendelova Univerzita v Brně, 2026