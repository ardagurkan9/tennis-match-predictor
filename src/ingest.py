# src/ingest.py
from pathlib import Path
import pandas as pd

def load_raw_matches(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_files = sorted(raw_dir.glob("atp_matches_*.csv"))

    frames = [pd.read_csv(path) for path in raw_files]
    return pd.concat(frames, ignore_index=True)