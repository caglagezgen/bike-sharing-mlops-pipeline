import json
import zipfile
from unittest.mock import patch

import pytest

from src.data.ingest import (
    _find_kaggle_cli,
    download_competition,
    ensure_kaggle_credentials,
)


class TestEnsureKaggleCredentials:
    def test_writes_json_from_env_vars(self, tmp_path, monkeypatch):
        """Env vars -> kaggle.json written with correct content and mode 600."""
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setenv("KAGGLE_KEY", "testkey")

        with patch("pathlib.Path.home", return_value=tmp_path):
            ensure_kaggle_credentials()

        kaggle_json = tmp_path / ".kaggle" / "kaggle.json"
        assert kaggle_json.exists()
        assert json.loads(kaggle_json.read_text()) == {
            "username": "testuser",
            "key": "testkey",
        }
        assert oct(kaggle_json.stat().st_mode)[-3:] == "600"

    def test_passes_when_json_file_already_exists(self, tmp_path, monkeypatch):
        """No env vars but kaggle.json present on disk -> no error raised."""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle.json").write_text('{"username":"u","key":"k"}')

        with patch("pathlib.Path.home", return_value=tmp_path):
            ensure_kaggle_credentials()  # must not raise

    def test_raises_when_no_credentials_available(self, tmp_path, monkeypatch):
        """No env vars, no file -> RuntimeError mentioning KAGGLE_USERNAME."""
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)

        with patch("pathlib.Path.home", return_value=tmp_path):
            with pytest.raises(RuntimeError, match="KAGGLE_USERNAME"):
                ensure_kaggle_credentials()


class TestFindKaggleCli:
    def test_raises_when_kaggle_not_on_path(self, monkeypatch):
        """All candidate paths missing -> RuntimeError with install hint."""
        with (
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.exists", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="kaggle CLI not found"):
                _find_kaggle_cli()


class TestDownloadCompetition:
    def test_skips_when_train_csv_already_exists(self, tmp_path):
        """train.csv present + force=False -> subprocess never called."""
        (tmp_path / "train.csv").touch()

        with (
            patch("src.data.ingest.ensure_kaggle_credentials"),
            patch("src.data.ingest.subprocess.run") as mock_run,
        ):
            download_competition("bike-sharing-demand", tmp_path, force=False)

        mock_run.assert_not_called()

    def test_calls_kaggle_cli_with_correct_args(self, tmp_path):
        """Kaggle CLI is invoked with competition slug, dest path, and --force."""
        (tmp_path / "train.csv").touch()

        with (
            patch("src.data.ingest.ensure_kaggle_credentials"),
            patch("src.data.ingest._find_kaggle_cli", return_value="/usr/bin/kaggle"),
            patch("src.data.ingest.subprocess.run") as mock_run,
            patch("src.data.ingest.validate"),
        ):
            download_competition("bike-sharing-demand", tmp_path, force=True)

        cmd = mock_run.call_args[0][0]
        assert "bike-sharing-demand" in cmd
        assert "--force" in cmd
        assert str(tmp_path) in cmd

    def test_extracts_zip_and_deletes_archive(self, tmp_path):
        """After subprocess runs, any ZIP in dest_dir is extracted and removed."""
        zip_path = tmp_path / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train.csv", "datetime,count\n2011-01-01 00:00:00,10\n")

        with (
            patch("src.data.ingest.ensure_kaggle_credentials"),
            patch("src.data.ingest._find_kaggle_cli", return_value="/usr/bin/kaggle"),
            patch("src.data.ingest.subprocess.run"),
            patch("src.data.ingest.validate"),
        ):
            download_competition("bike-sharing-demand", tmp_path, force=False)

        assert (tmp_path / "train.csv").exists()
        assert not zip_path.exists()

    def test_calls_validate_after_extraction(self, tmp_path):
        """validate() is called with the exact path of the extracted train.csv."""
        (tmp_path / "train.csv").touch()

        with (
            patch("src.data.ingest.ensure_kaggle_credentials"),
            patch("src.data.ingest._find_kaggle_cli", return_value="/usr/bin/kaggle"),
            patch("src.data.ingest.subprocess.run"),
            patch("src.data.ingest.validate") as mock_validate,
        ):
            download_competition("bike-sharing-demand", tmp_path, force=True)

        mock_validate.assert_called_once_with(tmp_path / "train.csv")

    def test_skips_validate_when_train_csv_not_produced(self, tmp_path):
        """If the download produces no train.csv, validate is never called."""
        with (
            patch("src.data.ingest.ensure_kaggle_credentials"),
            patch("src.data.ingest._find_kaggle_cli", return_value="/usr/bin/kaggle"),
            patch("src.data.ingest.subprocess.run"),
            patch("src.data.ingest.validate") as mock_validate,
        ):
            download_competition("bike-sharing-demand", tmp_path, force=True)

        mock_validate.assert_not_called()
