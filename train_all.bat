@echo off
echo ========================================
echo  TRENOVANI VSECH MODELU
echo ========================================

echo.
echo [1/16] total_goals...
python src/model_count_glm.py --target total_goals --prefix goals --family poisson --save_artifacts

echo.
echo [2/16] FTHG (home goals)...
python src/model_count_glm.py --target FTHG --prefix goals --family poisson --save_artifacts

echo.
echo [3/16] FTAG (away goals)...
python src/model_count_glm.py --target FTAG --prefix goals --family poisson --save_artifacts

echo.
echo [4/16] total_corners...
python src/model_count_glm.py --target total_corners --prefix corners --family negbin --save_artifacts

echo.
echo [5/16] HC (home corners)...
python src/model_count_glm.py --target HC --prefix corners --family negbin --save_artifacts

echo.
echo [6/16] AC (away corners)...
python src/model_count_glm.py --target AC --prefix corners --family negbin --save_artifacts

echo.
echo [7/16] total_fouls...
python src/model_count_glm.py --target total_fouls --prefix fouls --family negbin --save_artifacts

echo.
echo [8/16] HF (home fouls)...
python src/model_count_glm.py --target HF --prefix fouls --family negbin --save_artifacts

echo.
echo [9/16] AF (away fouls)...
python src/model_count_glm.py --target AF --prefix fouls --family negbin --save_artifacts

echo.
echo [10/16] total_cards...
python src/model_count_glm.py --target total_cards --prefix yellow --family negbin --save_artifacts

echo.
echo [11/16] HY (home yellow cards)...
python src/model_count_glm.py --target HY --prefix yellow --family negbin --save_artifacts

echo.
echo [12/16] AY (away yellow cards)...
python src/model_count_glm.py --target AY --prefix yellow --family negbin --save_artifacts

echo.
echo [13/16] total_shots_on_target...
python src/model_count_glm.py --target total_shots_on_target --prefix shotsot --family poisson --save_artifacts

echo.
echo [14/16] HST (home shots on target)...
python src/model_count_glm.py --target HST --prefix shotsot --family poisson --save_artifacts

echo.
echo [15/16] AST (away shots on target)...
python src/model_count_glm.py --target AST --prefix shotsot --family poisson --save_artifacts

echo.
echo [16/16] 1X2 (vysledek zapasu)...
python src/model_1X2.py --save_artifacts --calibrate

echo.
echo ========================================
echo  HOTOVO! Vsechny modely natrenovat.
echo ========================================
pause