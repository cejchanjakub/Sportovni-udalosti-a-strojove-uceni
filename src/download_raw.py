from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


@dataclass(frozen=True)
class SeasonSpec:
    label: str          # e.g. "2024-25"
    code: str           # e.g. "2425" (football-data path)
    out_name: str       # e.g. "EPL_2024-25.csv"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return sha256_bytes(path.read_bytes())


def download_csv(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def append_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log_path = project_root / "docs" / "download_log.csv"

    # Premier League code on football-data is "E0"
    league_code = "E0"
    base_url = "https://www.football-data.co.uk/mmz4281"

    # 10 sezon + aktualni 2025/26 (uprav podle sveho „okna“)
    seasons = [
        SeasonSpec("2016-17", "1617", "EPL_2016-17.csv"),
        SeasonSpec("2017-18", "1718", "EPL_2017-18.csv"),
        SeasonSpec("2018-19", "1819", "EPL_2018-19.csv"),
        SeasonSpec("2019-20", "1920", "EPL_2019-20.csv"),
        SeasonSpec("2020-21", "2021", "EPL_2020-21.csv"),
        SeasonSpec("2021-22", "2122", "EPL_2021-22.csv"),
        SeasonSpec("2022-23", "2223", "EPL_2022-23.csv"),
        SeasonSpec("2023-24", "2324", "EPL_2023-24.csv"),
        SeasonSpec("2024-25", "2425", "EPL_2024-25.csv"),
        SeasonSpec("2025-26", "2526", "EPL_2025-26.csv"),  # živá sezóna
    ]

    for s in seasons:
        url = f"{base_url}/{s.code}/{league_code}.csv"
        out_path = raw_dir / s.out_name

        ts = datetime.now().isoformat(timespec="seconds")

        try:
            new_bytes = download_csv(url)
            new_hash = sha256_bytes(new_bytes)
            old_hash = file_sha256(out_path)

            changed = (old_hash != new_hash)
            if changed:
                out_path.write_bytes(new_bytes)

            append_log(log_path, {
                "timestamp": ts,
                "season": s.label,
                "url": url,
                "file": str(out_path),
                "changed": str(changed),
                "old_hash": old_hash or "",
                "new_hash": new_hash,
                "bytes": str(len(new_bytes)),
            })

            print(f"[OK] {s.label} -> {out_path.name} | changed={changed}")

        except Exception as e:
            append_log(log_path, {
                "timestamp": ts,
                "season": s.label,
                "url": url,
                "file": str(out_path),
                "changed": "ERROR",
                "old_hash": file_sha256(out_path) or "",
                "new_hash": "",
                "bytes": "",
            })
            print(f"[FAIL] {s.label} | {e}")

    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
