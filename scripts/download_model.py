"""Download the released production model with an optional SHA-256 check."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://github.com/ardagurkan9/tennis-match-predictor/releases/latest/"
    "download/lightgbm.pkl"
)
DEFAULT_OUTPUT = Path("models/advanced/lightgbm.pkl")


def sha256(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(url: str, output: Path, expected_sha256: str | None = None) -> Path:
    """Download atomically, keeping an existing artifact intact on failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=output.parent) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with temporary_path.open("wb") as destination:
                shutil.copyfileobj(response, destination)
        actual_sha256 = sha256(temporary_path)
        if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sha256", help="Optional expected SHA-256 digest")
    args = parser.parse_args()
    path = download_model(args.url, args.output, args.sha256)
    print(f"Model saved to {path} (sha256={sha256(path)})")


if __name__ == "__main__":
    main()

