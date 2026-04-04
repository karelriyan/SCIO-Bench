"""
Phase 1 — Dataset Download
Downloads the Kaggle Solar Power Generation dataset to data/raw/.

Dataset: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
Author: Ani Kannal (2020)
Files:
  Plant_1_Generation_Data.csv
  Plant_1_Weather_Sensor_Data.csv
  Plant_2_Generation_Data.csv
  Plant_2_Weather_Sensor_Data.csv
"""

import os
import pathlib
import json
import stat
import shutil
import subprocess

DATASET_SLUG = "anikannal/solar-power-generation-data"
DOWNLOAD_DIR = pathlib.Path("data/raw")
EXPECTED_FILES = [
    "Plant_1_Generation_Data.csv",
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Generation_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv",
]


def _ensure_kaggle_credentials(api_key: str | None = None, username: str | None = None) -> None:
    """
    Write kaggle.json to ~/.kaggle/ if not already present.
    Accepts explicit api_key and username, or falls back to env vars, or existing file.
    """
    kaggle_dir = pathlib.Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        return  # Already configured

    if api_key is None:
        api_key = os.environ.get("KAGGLE_KEY", "")
    if username is None:
        username = os.environ.get("KAGGLE_USERNAME", "kgat")

    if not api_key:
        raise EnvironmentError(
            "Kaggle API key not found. Please either:\n"
            "  1. Run setup_kaggle.py, or\n"
            "  2. Set KAGGLE_KEY environment variable, or\n"
            "  3. Place kaggle.json in ~/.kaggle/"
        )

    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": api_key}))
    kaggle_json.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"Kaggle credentials written to {kaggle_json}")


def _files_already_downloaded(dest: pathlib.Path) -> bool:
    """Return True if all expected CSV files exist in dest."""
    return all((dest / f).exists() for f in EXPECTED_FILES)


def download_kaggle_dataset(
    dest: str | pathlib.Path = "data/raw",
    api_key: str | None = None,
    username: str | None = None,
    force: bool = False,
) -> pathlib.Path:
    """
    Download and extract the Kaggle Solar Power Generation dataset.

    Args:
        dest:     Destination directory (default: data/raw/).
        api_key:  Kaggle API key (KGAT_ token). Falls back to env/~/.kaggle/kaggle.json.
        username: Kaggle username. Falls back to 'kgat' or env var.
        force:    Re-download even if files already exist.

    Returns:
        pathlib.Path of the destination directory.
    """
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not force and _files_already_downloaded(dest):
        print(f"[download] Dataset already present in {dest}/ — skipping download.")
        return dest

    _ensure_kaggle_credentials(api_key=api_key, username=username)

    print(f"[download] Downloading {DATASET_SLUG} to {dest}/ …")

    # Locate the kaggle CLI binary (installed in the active venv)
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        raise EnvironmentError(
            "[download] 'kaggle' binary not found in PATH.\n"
            "Run: pip install kaggle"
        )

    result = subprocess.run(
        [kaggle_bin, "datasets", "download",
         "-d", DATASET_SLUG,
         "--path", str(dest),
         "--unzip"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[download] Kaggle CLI error (exit {result.returncode}):\n"
            f"{result.stderr}\n"
            "Tip: verify your Kaggle credentials at https://www.kaggle.com/settings\n"
            f"  Username in ~/.kaggle/kaggle.json may need updating."
        )

    if result.stdout:
        print(result.stdout)
    print(f"[download] Download complete.")

    # Verify expected files
    missing = [f for f in EXPECTED_FILES if not (dest / f).exists()]
    if missing:
        raise FileNotFoundError(f"[download] Missing files after extraction: {missing}")

    print(f"[download] ✓ All 4 CSV files extracted to {dest}/")
    return dest


if __name__ == "__main__":
    # Quick test: python -m src.data.download
    download_kaggle_dataset()
