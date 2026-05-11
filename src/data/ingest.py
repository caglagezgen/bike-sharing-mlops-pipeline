import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.config import RAW_DIR, ensure_dirs
from src.data.validate import DataQualityError, validate


def ensure_kaggle_credentials() -> None:
    # Prefer environment variables (used in GitHub Actions via secrets).
    # Fall back to ~/.kaggle/kaggle.json for local runs.
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if user and key:
        # Write the credentials file so the kaggle CLI can find them.
        # chmod 600 is required by the kaggle CLI — it refuses to read world-readable files.
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": user, "key": key}))
        os.chmod(kaggle_json, 0o600)
        return

    if not kaggle_json.exists():
        raise RuntimeError(
            "KAGGLE_USERNAME/KAGGLE_KEY not set and ~/.kaggle/kaggle.json not found"
        )


def _find_kaggle_cli() -> str:
    """Locate the kaggle CLI regardless of which Python is running the project."""
    candidates = [
        shutil.which("kaggle"),
        "/opt/homebrew/bin/kaggle",
        str(Path.home() / ".local/bin/kaggle"),
        "/usr/local/bin/kaggle",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    raise RuntimeError(
        "kaggle CLI not found. Install it with: pip install kaggle\n"
        "Then make sure ~/.kaggle/kaggle.json exists."
    )


def download_competition(competition: str, dest_dir: Path, force: bool) -> None:
    ensure_kaggle_credentials()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Skip if data is already present to avoid unnecessary API calls and quota usage.
    if (dest_dir / "train.csv").exists() and not force:
        print(f"Data already present at {dest_dir} — skipping (use --force to re-download).")
        return

    kaggle_cli = _find_kaggle_cli()
    cmd = [kaggle_cli, "competitions", "download", "-c", competition, "-p", str(dest_dir)]
    if force:
        cmd.append("--force")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Kaggle downloads a single ZIP; extract it and remove the archive.
    for zip_path in dest_dir.glob("*.zip"):
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()

    # Validate the downloaded CSV immediately after extraction so a corrupt
    # or partial download is caught before it reaches preprocessing or training.
    train_csv = dest_dir / "train.csv"
    if train_csv.exists():
        validate(train_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle competition data")
    parser.add_argument(
        "--competition", default="bike-sharing-demand", help="Kaggle competition slug"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    ensure_dirs()
    try:
        download_competition(args.competition, RAW_DIR, args.force)
    except DataQualityError as exc:
        print(f"[ingest] Aborting — data quality check failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
