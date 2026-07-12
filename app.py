
from __future__ import annotations

import io
import json
import os
import random
import sys
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import sounddevice as sd
import torch
from demucs.apply import apply_model
from demucs.audio import AudioFile, convert_audio
from demucs.pretrained import get_model
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import demucs.apply as demucs_apply
import tqdm as tqdm_mod

from downloader import (
    download_track,
    empty_manifest,
    fetch_playlist_manifest,
    fetch_video_entry,
)
from runtime_env import data_dir, ensure_ffmpeg_on_path

ROOT = data_dir()
SONGSDIR = ROOT / "songs"
YT_DOWNLOAD_DIR = SONGSDIR / "yt_downloads"
SETTINGS_DIR = ROOT / "defaults"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".aiff", ".aif"}
MODEL_NAME = "htdemucs_6s"
SNIPPET_DURATION = 10.0
# Reveal least-identifiable stems first; vocals last.
STEM_ORDER = ("drums", "bass", "other", "guitar", "piano", "vocals")

# Overall prepare pipeline weights (download → separate → finish).
PROGRESS_DOWNLOAD = (0.0, 0.30)
PROGRESS_SEPARATE = (0.30, 0.95)
PROGRESS_FINISH = (0.95, 1.0)


def _map_progress(frac: float | None, start: float, end: float) -> float | None:
    if frac is None:
        return None
    return start + max(0.0, min(1.0, frac)) * (end - start)


def default_settings() -> dict[str, Any]:
    return {
        "mode": "local",
        "playlist": None,
        "played_entry_ids": [],
    }


def load_settings() -> dict[str, Any]:
    data = default_settings()
    if not SETTINGS_PATH.is_file():
        return data
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return data
    if not isinstance(raw, dict):
        return data
    mode = raw.get("mode")
    if mode in ("local", "playlist"):
        data["mode"] = mode
    playlist = raw.get("playlist")
    if isinstance(playlist, dict) and isinstance(playlist.get("entries"), list):
        data["playlist"] = playlist
    played = raw.get("played_entry_ids")
    if isinstance(played, list):
        data["played_entry_ids"] = [str(x) for x in played]
    return data


def save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": settings.get("mode", "local"),
        "playlist": settings.get("playlist"),
        "played_entry_ids": list(settings.get("played_entry_ids") or []),
    }
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class _ProgressTqdm(tqdm_mod.tqdm):
    """tqdm stand-in that forwards Demucs chunk progress to a callback."""

    def __init__(self, *args, progress_cb=None, **kwargs):
        # Keep enabled so iteration runs the progress path; swallow terminal output.
        kwargs["disable"] = False
        kwargs["file"] = io.StringIO()
        kwargs["dynamic_ncols"] = False
        kwargs["mininterval"] = 0
        kwargs["miniters"] = 1
        self._progress_cb = progress_cb
        super().__init__(*args, **kwargs)
        self._emit()

    def _emit(self):
        if not self._progress_cb:
            return
        total = self.total or 0
        if total <= 0:
            self._progress_cb(None)
            return
        self._progress_cb(min(self.n / total, 1.0))

    def update(self, n=1):
        result = super().update(n)
        self._emit()
        return result

    def close(self):
        self._emit()
        return super().close()


def _hook_demucs_tqdm(progress_cb):
    """
    Demucs calls ``tqdm.tqdm(...)`` (module attribute), so patch the class
    on the imported module rather than replacing the module object.
    """
    original_cls = demucs_apply.tqdm.tqdm

    def factory(*args, **kwargs):
        return _ProgressTqdm(*args, progress_cb=progress_cb, **kwargs)

    demucs_apply.tqdm.tqdm = factory
    return original_cls


def _unhook_demucs_tqdm(original_cls):
    demucs_apply.tqdm.tqdm = original_cls


def list_songs(directory: Path = SONGSDIR) -> list[Path]:
    if not directory.is_dir():
        return []
    songs: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS and not path.name.startswith("."):
            songs.append(path)
    return songs


def song_label(path: Path, root: Path = SONGSDIR) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return str(rel.with_suffix(""))


def normalize_guess(text: str) -> str:
    return " ".join(text.strip().lower().split())


def find_consensus_window(separated_stems: dict[str, torch.Tensor], sr: int, snippet_duration: float = 10.0):
    """Find a synchronized window where all stems have high energy."""
    rms_curves = []
    hop_length = 512
    frame_length = 2048

    for waveform in separated_stems.values():
        audio_np = waveform.detach().cpu().numpy()
        if audio_np.ndim > 1 and audio_np.shape[0] > 1:
            audio_np = np.mean(audio_np, axis=0)
        else:
            audio_np = audio_np.flatten()

        rms = librosa.feature.rms(y=audio_np, frame_length=frame_length, hop_length=hop_length)[0]
        rms_min, rms_max = float(rms.min()), float(rms.max())
        if rms_max > rms_min:
            rms_norm = (rms - rms_min) / (rms_max - rms_min)
        else:
            rms_norm = rms
        rms_curves.append(rms_norm)

    min_frames = min(len(r) for r in rms_curves)
    rms_curves = [r[:min_frames] for r in rms_curves]
    combined_energy = np.exp(np.mean(np.log(np.asarray(rms_curves) + 1e-5), axis=0))

    frames_per_second = sr / hop_length
    snippet_frames = int(snippet_duration * frames_per_second)

    if min_frames <= snippet_frames:
        return 0.0, float(min_frames / frames_per_second)

    window_energies = np.convolve(combined_energy, np.ones(snippet_frames), mode="valid")
    best_start_frame = int(np.argmax(window_energies))
    start_time = best_start_frame * hop_length / sr
    end_time = start_time + snippet_duration
    return start_time, end_time


def load_audio_for_model(track: Path, model) -> torch.Tensor:
    errors: dict[str, str] = {}
    wav = None
    try:
        wav = AudioFile(str(track)).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels,
        )
    except FileNotFoundError:
        errors["ffmpeg"] = "FFmpeg is not installed."
    except Exception as exc:  # noqa: BLE001 — surface any decode failure below
        errors["ffmpeg"] = str(exc)

    if wav is None:
        try:
            import torchaudio as ta

            wav, sr = ta.load(str(track))
            wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
        except Exception as exc:  # noqa: BLE001
            errors["torchaudio"] = str(exc)

    if wav is None:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items()) or "unsupported format"
        raise RuntimeError(f"Could not load audio ({detail})")
    return wav


def separate_song(song_path: Path, status_cb=None, progress_cb=None) -> dict:
    """Run Demucs and return clipped stem waveforms for the best consensus window."""
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    def emit(msg: str, frac: float | None = None):
        if progress_cb:
            progress_cb(msg, frac)
        elif status_cb:
            status_cb(msg)

    emit("Loading model…", None)

    model = get_model(MODEL_NAME)
    model.to(device)
    model.eval()

    emit("Loading audio…", None)
    wav = load_audio_for_model(song_path, model)

    ref = wav.mean(0)
    wav = wav - ref.mean()
    wav = wav / (ref.std() + 1e-8)

    emit("Separating stems…", 0.0)

    def on_demucs_frac(frac: float | None):
        if frac is None:
            emit("Separating stems…", None)
        else:
            emit(f"Separating stems… {int(frac * 100)}%", frac)

    original_cls = _hook_demucs_tqdm(on_demucs_frac)
    try:
        sources = apply_model(
            model,
            wav[None],
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True,
        )[0]
    finally:
        _unhook_demucs_tqdm(original_cls)

    sources = sources * (ref.std() + 1e-8)
    sources = sources + ref.mean()

    separated = {name: sources[i].detach().cpu() for i, name in enumerate(model.sources)}
    sr = int(model.samplerate)

    emit("Finding best window…", 1.0)
    start_time, end_time = find_consensus_window(separated, sr, SNIPPET_DURATION)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    order = [s for s in STEM_ORDER if s in separated]
    order.extend(s for s in separated if s not in order)

    clipped: dict[str, np.ndarray] = {}
    for name in order:
        stem = separated[name][:, start_sample:end_sample].numpy()
        clipped[name] = stem.astype(np.float32, copy=False)

    return {
        "path": song_path,
        "label": song_label(song_path),
        "sr": sr,
        "stem_order": order,
        "stems": clipped,
        "window": (start_time, end_time),
    }


def mix_stems(stems: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Sum selected stems to shape [samples, channels] for sounddevice."""
    if not names:
        raise ValueError("no stems to mix")
    mix = None
    for name in names:
        wav = stems[name]
        if mix is None:
            mix = wav.astype(np.float32, copy=True)
        else:
            n = min(mix.shape[-1], wav.shape[-1])
            mix = mix[..., :n] + wav[..., :n]
    assert mix is not None
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak
    # sounddevice expects [samples, channels]
    if mix.ndim == 1:
        return mix[:, None]
    return np.ascontiguousarray(mix.T)


class PrepareWorker(QThread):
    status = pyqtSignal(str)
    progress = pyqtSignal(object)  # float 0–1, or None for indeterminate
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        song_path: Path | None = None,
        playlist_entry: dict[str, Any] | None = None,
        answer_label: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.song_path = song_path
        self.playlist_entry = playlist_entry
        self.answer_label = answer_label

    def _emit(self, msg: str, frac: float | None):
        self.status.emit(msg)
        self.progress.emit(frac)

    def run(self):
        try:
            path = self.song_path
            if self.playlist_entry is not None:
                lo, hi = PROGRESS_DOWNLOAD

                def download_progress(msg: str, frac: float | None):
                    self._emit(msg, _map_progress(frac, lo, hi))

                path = download_track(
                    self.playlist_entry,
                    YT_DOWNLOAD_DIR,
                    progress_cb=download_progress,
                )
            assert path is not None

            lo, hi = PROGRESS_SEPARATE

            def separate_progress(msg: str, frac: float | None):
                self._emit(msg, _map_progress(frac, lo, hi))

            result = separate_song(path, progress_cb=separate_progress)
            if self.answer_label:
                result["label"] = self.answer_label
            self._emit("Ready", PROGRESS_FINISH[1])
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ManifestWorker(QThread):
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, playlist_url: str, parent=None):
        super().__init__(parent)
        self.playlist_url = playlist_url

    def run(self):
        try:
            manifest = fetch_playlist_manifest(self.playlist_url, status_cb=self.status.emit)
            self.finished_ok.emit(manifest)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ResolveLinkWorker(QThread):
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self.url = url

    def run(self):
        try:
            entry = fetch_video_entry(self.url, status_cb=self.status.emit)
            self.finished_ok.emit(entry)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class SettingsDialog(QDialog):
    """Persisted settings: source mode, YouTube playlist, edit manifest."""

    def __init__(self, settings: dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.settings = {
            "mode": settings.get("mode", "local"),
            "playlist": settings.get("playlist"),
            "played_entry_ids": list(settings.get("played_entry_ids") or []),
        }
        self._worker: ManifestWorker | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        mode_label = QLabel("Song source")
        mode_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(mode_label)

        mode_row = QHBoxLayout()
        self.local_radio = QRadioButton("Local files")
        self.yt_radio = QRadioButton("YouTube playlist")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.local_radio)
        self._mode_group.addButton(self.yt_radio)
        if self.settings["mode"] == "playlist":
            self.yt_radio.setChecked(True)
        else:
            self.local_radio.setChecked(True)
        self.local_radio.toggled.connect(self._sync_youtube_panel)
        mode_row.addWidget(self.local_radio)
        mode_row.addWidget(self.yt_radio)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.yt_panel = QWidget()
        yt_layout = QVBoxLayout(self.yt_panel)
        yt_layout.setContentsMargins(0, 8, 0, 0)
        yt_layout.setSpacing(8)

        yt_layout.addWidget(QLabel("Playlist URL"))
        self.url_input = QLineEdit()
        playlist = self.settings.get("playlist") or {}
        self.url_input.setText(str(playlist.get("url") or ""))
        self.url_input.setPlaceholderText("https://www.youtube.com/playlist?list=…")
        yt_layout.addWidget(self.url_input)

        yt_btns = QHBoxLayout()
        self.clearcache = QPushButton("Clear cache")
        self.clearcache.clicked.connect(self._clearcache)
        self.load_btn = QPushButton("Load playlist")
        self.load_btn.clicked.connect(self._load_playlist)
        self.empty_btn = QPushButton("Empty list")
        self.empty_btn.clicked.connect(self._start_empty)
        self.edit_btn = QPushButton("Edit manifest…")
        self.edit_btn.clicked.connect(self._edit_manifest)
        yt_btns.addWidget(self.load_btn)
        yt_btns.addWidget(self.empty_btn)
        yt_btns.addWidget(self.edit_btn)
        yt_btns.addWidget(self.clearcache)
        yt_layout.addLayout(yt_btns)

        self.yt_status = QLabel("")
        self.yt_status.setWordWrap(True)
        self.yt_status.setStyleSheet("color:#8b949e;")
        yt_layout.addWidget(self.yt_status)

        layout.addWidget(self.yt_panel)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._sync_youtube_panel()
        self._refresh_yt_status()

    def _sync_youtube_panel(self):
        self.yt_panel.setEnabled(self.yt_radio.isChecked())

    def _refresh_yt_status(self):
        playlist = self.settings.get("playlist")
        if not playlist:
            self.yt_status.setText("No playlist loaded yet.")
            return
        n = len(playlist.get("entries") or [])
        self.yt_status.setText(f"“{playlist.get('title') or 'Playlist'}” · {n} tracks")

    def _start_empty(self):
        self.settings["playlist"] = empty_manifest()
        self.settings["played_entry_ids"] = []
        self.url_input.clear()
        self._refresh_yt_status()

    def _clearcache(self):
        os.remove(SONGSDIR)

    def _load_playlist(self):
        url = self.url_input.text().strip()
        url = url.replace("view", "playlist")
        if not url:
            QMessageBox.warning(self, "Missing URL", "Paste a YouTube playlist URL first.")
            return
        self.load_btn.setEnabled(False)
        self.yt_status.setText("Fetching playlist manifest…")
        self._worker = ManifestWorker(url, self)
        self._worker.status.connect(self.yt_status.setText)
        self._worker.finished_ok.connect(self._on_manifest)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_manifest(self, manifest: dict):
        self.load_btn.setEnabled(True)
        self.settings["playlist"] = manifest
        self.settings["played_entry_ids"] = []
        self.url_input.setText(manifest.get("url") or self.url_input.text())
        self._refresh_yt_status()

    def _on_failed(self, message: str):
        self.load_btn.setEnabled(True)
        self.yt_status.setText("Failed to load playlist.")
        QMessageBox.critical(self, "Playlist failed", message)

    def _edit_manifest(self):
        if not self.settings.get("playlist"):
            self.settings["playlist"] = empty_manifest()
        before = {str(e["id"]) for e in self.settings["playlist"]["entries"]}
        dlg = ManifestEditorDialog(self.settings["playlist"], self)
        dlg.exec()
        after = {str(e["id"]) for e in self.settings["playlist"]["entries"]}
        played = set(self.settings.get("played_entry_ids") or [])
        self.settings["played_entry_ids"] = list(played - (before - after))
        self._refresh_yt_status()

    def _save(self):
        if self.yt_radio.isChecked():
            self.settings["mode"] = "playlist"
            if not self.settings.get("playlist"):
                QMessageBox.warning(
                    self,
                    "No playlist",
                    "Load a playlist, start an empty list, or switch to Local.",
                )
                return
        else:
            self.settings["mode"] = "local"
        self.accept()


class PreparingDialog(QDialog):
    """Progress-only prepare UI — never shows the song title."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preparing game")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        title = QLabel("Preparing round")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        hint = QLabel("Hang tight — this can take a minute.")
        hint.setStyleSheet("color:#8b949e;")
        layout.addWidget(hint)

        self.status = QLabel("Starting…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setFormat("%p%")
        self.bar.setTextVisible(True)
        layout.addWidget(self.bar)

    def set_status(self, text: str):
        self.status.setText(text)

    def set_progress(self, frac: object):
        """frac: float 0–1, or None for indeterminate."""
        if frac is None:
            self.bar.setRange(0, 0)
            self.bar.setFormat("")
            return
        value = int(max(0.0, min(1.0, float(frac))) * 1000)
        if self.bar.maximum() != 1000:
            self.bar.setRange(0, 1000)
            self.bar.setFormat("%p%")
        self.bar.setValue(value)


class ManifestEditorDialog(QDialog):
    """Pop-out editor for the YouTube playlist manifest."""

    def __init__(self, manifest: dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit manifest")
        self.setModal(True)
        self.resize(520, 480)
        self.manifest = manifest
        self._resolve_worker: ResolveLinkWorker | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.heading = QLabel()
        self.heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(self.heading)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.list, stretch=1)

        add_row = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste a YouTube link…")
        self.url_input.returnPressed.connect(self._add_url)
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self._add_url)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        add_row.addWidget(self.url_input, stretch=1)
        add_row.addWidget(self.add_btn)
        add_row.addWidget(self.remove_btn)
        layout.addLayout(add_row)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#8b949e;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)

        self._refresh()

    def _refresh(self):
        self.list.clear()
        for entry in self.manifest["entries"]:
            item = QListWidgetItem(entry["title"])
            item.setData(Qt.ItemDataRole.UserRole, entry["id"])
            item.setToolTip(entry.get("url", ""))
            self.list.addItem(item)
        n = len(self.manifest["entries"])
        self.heading.setText(f"{self.manifest.get('title') or 'Manifest'} · {n} tracks")

    def _add_url(self):
        url = self.url_input.text().strip()
        if not url:
            return
        if self._resolve_worker is not None and self._resolve_worker.isRunning():
            self.status.setText("Still resolving the previous link…")
            return
        self.add_btn.setEnabled(False)
        self.url_input.setEnabled(False)
        self.status.setText("Resolving link…")
        self._resolve_worker = ResolveLinkWorker(url, self)
        self._resolve_worker.status.connect(self.status.setText)
        self._resolve_worker.finished_ok.connect(self._on_resolved)
        self._resolve_worker.failed.connect(self._on_failed)
        self._resolve_worker.start()

    def _on_resolved(self, entry: dict):
        self.add_btn.setEnabled(True)
        self.url_input.setEnabled(True)
        existing = {str(e["id"]) for e in self.manifest["entries"]}
        if str(entry["id"]) in existing:
            self.status.setText("Already in the manifest.")
            self.url_input.clear()
            return
        self.manifest["entries"].append(entry)
        self.url_input.clear()
        self.status.setText("Track added.")
        self._refresh()

    def _on_failed(self, message: str):
        self.add_btn.setEnabled(True)
        self.url_input.setEnabled(True)
        self.status.setText("Could not add that link.")
        QMessageBox.critical(self, "Add failed", message)

    def _remove_selected(self):
        selected = self.list.selectedItems()
        if not selected:
            self.status.setText("Select one or more tracks to remove.")
            return
        remove_ids = {str(item.data(Qt.ItemDataRole.UserRole)) for item in selected}
        before = len(self.manifest["entries"])
        self.manifest["entries"] = [
            e for e in self.manifest["entries"] if str(e["id"]) not in remove_ids
        ]
        self.status.setText(f"Removed {before - len(self.manifest['entries'])} track(s).")
        self._refresh()


class AttemptRow(QWidget):
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.badge = QLabel(f"{index + 1}")
        self.badge.setFixedWidth(28)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setStyleSheet(
            "background:#1f6feb; color:white; border-radius:6px; padding:6px 0; font-weight:600;"
        )

        self.text = QLabel("—")
        self.text.setStyleSheet(
            "background:#161b22; color:#c9d1d9; border:1px solid #30363d; "
            "border-radius:8px; padding:10px 12px;"
        )
        self.text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout.addWidget(self.badge)
        layout.addWidget(self.text)

    def set_guess(self, guess: str, correct: bool | None):
        self.text.setText(guess)
        if correct is True:
            self.text.setStyleSheet(
                "background:#238636; color:white; border:1px solid #2ea043; "
                "border-radius:8px; padding:10px 12px; font-weight:600;"
            )
            self.badge.setStyleSheet(
                "background:#238636; color:white; border-radius:6px; padding:6px 0; font-weight:600;"
            )
        elif correct is False:
            self.text.setStyleSheet(
                "background:#21262d; color:#f85149; border:1px solid #da3633; "
                "border-radius:8px; padding:10px 12px;"
            )
            self.badge.setStyleSheet(
                "background:#da3633; color:white; border-radius:6px; padding:6px 0; font-weight:600;"
            )

    def set_forfeit(self):
        self.text.setText("Forfeit")
        self.text.setStyleSheet(
            "background:#21262d; color:#d29922; border:1px solid #9e6a03; "
            "border-radius:8px; padding:10px 12px;"
        )
        self.badge.setStyleSheet(
            "background:#9e6a03; color:white; border-radius:6px; padding:6px 0; font-weight:600;"
        )


class BandleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bandle-local")
        self.resize(720, 780)

        self.settings = load_settings()
        self.mode: str = self.settings.get("mode", "local")
        self.playlist_manifest: dict[str, Any] | None = self.settings.get("playlist")
        self._played_entry_ids: set[str] = set(self.settings.get("played_entry_ids") or [])

        self.songs = list_songs()
        self.song_by_label = {song_label(p): p for p in self.songs}
        self.labels = sorted(self.song_by_label.keys(), key=str.lower)

        self.round: dict | None = None
        self.revealed = 0
        self.guesses: list[str] = []
        self.game_over = False
        self._current_mix: np.ndarray | None = None
        self._prepare_dialog: PreparingDialog | None = None
        self._worker: PrepareWorker | None = None
        self._play_lock = threading.Lock()

        self._build_ui()
        self._set_controls_enabled(False)
        self._apply_mode_labels()

    def _persist_settings(self):
        self.settings = {
            "mode": self.mode,
            "playlist": self.playlist_manifest,
            "played_entry_ids": sorted(self._played_entry_ids),
        }
        save_settings(self.settings)

    def _apply_mode_labels(self):
        playlist = self.mode == "playlist" and self.playlist_manifest is not None

        if playlist:
            entries = self.playlist_manifest["entries"]
            seen: set[str] = set()
            labels: list[str] = []
            for entry in entries:
                title = entry["title"]
                if title in seen:
                    continue
                seen.add(title)
                labels.append(title)
            self.labels = sorted(labels, key=str.lower)
            n = len(entries)
            title = self.playlist_manifest["title"]
            self.status_label.setText(f"YouTube · {n} tracks · “{title}”")
        else:
            self.songs = list_songs()
            self.song_by_label = {song_label(p): p for p in self.songs}
            self.labels = sorted(self.song_by_label.keys(), key=str.lower)
            if not self.songs:
                self.status_label.setText(f"No songs in {SONGSDIR}. Add audio, then New Game.")
            else:
                self.status_label.setText(f"Local · {len(self.songs)} songs")

        self.subtitle.setText("Guess the song as stems stack after each miss.")
        self._filter_suggestions(self.guess_input.text())
        self._update_play_next_label()

    def _update_play_next_label(self):
        if self.round is None or self.game_over:
            self.play_next_btn.setText("Play next")
            return
        guess_num = len(self.guesses) + 1
        self.play_next_btn.setText(f"Play next (forfeit #{guess_num})")

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root.setStyleSheet(
            """
            QWidget { background: #0d1117; color: #e6edf3; font-size: 14px; }
            QLineEdit {
                background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 10px 12px; selection-background-color: #1f6feb;
            }
            QListWidget {
                background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                outline: none;
            }
            QListWidget::item { padding: 8px 10px; }
            QListWidget::item:selected { background: #1f6feb; color: white; }
            QPushButton {
                background: #21262d; border: 1px solid #30363d; border-radius: 8px;
                padding: 10px 14px; font-weight: 600;
            }
            QPushButton:hover { background: #30363d; }
            QPushButton:disabled { color: #6e7681; background: #161b22; }
            QPushButton#primary {
                background: #1f6feb; border-color: #1f6feb; color: white;
            }
            QPushButton#primary:hover { background: #388bfd; }
            QProgressBar {
                background: #161b22; border: 1px solid #30363d; border-radius: 6px;
                text-align: center; height: 18px;
            }
            QProgressBar::chunk { background: #1f6feb; border-radius: 5px; }
            """
        )

        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        top = QHBoxLayout()
        top.addStretch(1)
        header = QLabel("BANDLE")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            "font-size: 34px; font-weight: 800; letter-spacing: 4px; color: #58a6ff;"
        )
        top.addWidget(header)
        top.addStretch(1)
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        top.addWidget(self.settings_btn)
        layout.addLayout(top)

        self.subtitle = QLabel("Guess the song as stems stack after each miss.")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet("color:#8b949e;")
        layout.addWidget(self.subtitle)

        self.attempts_host = QVBoxLayout()
        self.attempts_host.setSpacing(8)
        attempts_wrap = QWidget()
        attempts_wrap.setLayout(self.attempts_host)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(attempts_wrap)
        scroll.setMinimumHeight(220)
        layout.addWidget(scroll)
        self.attempt_rows: list[AttemptRow] = []

        self.guess_input = QLineEdit()
        self.guess_input.setPlaceholderText("Type a song title…")
        self.guess_input.textChanged.connect(self._filter_suggestions)
        self.guess_input.returnPressed.connect(self._submit_from_input)
        layout.addWidget(self.guess_input)

        self.suggestions = QListWidget()
        self.suggestions.setMaximumHeight(160)
        self.suggestions.itemClicked.connect(self._pick_suggestion)
        self.suggestions.itemActivated.connect(self._pick_suggestion)
        layout.addWidget(self.suggestions)
        self._filter_suggestions("")

        btn_row = QHBoxLayout()
        self.play_next_btn = QPushButton("Play next")
        self.replay_btn = QPushButton("Replay")
        self.forfeit_btn = QPushButton("Forfeit game")
        self.submit_btn = QPushButton("Guess")
        self.submit_btn.setObjectName("primary")
        self.new_btn = QPushButton("New Game")
        self.new_btn.setObjectName("primary")

        self.play_next_btn.clicked.connect(self.forfeit_guess)
        self.replay_btn.clicked.connect(self.replay_current)
        self.forfeit_btn.clicked.connect(self.forfeit_game)
        self.submit_btn.clicked.connect(self._submit_from_input)
        self.new_btn.clicked.connect(self.start_new_game)

        for btn in (
            self.play_next_btn,
            self.replay_btn,
            self.forfeit_btn,
            self.submit_btn,
            self.new_btn,
        ):
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#8b949e;")
        layout.addWidget(self.status_label)

    def open_settings(self):
        snapshot = {
            "mode": self.mode,
            "playlist": self.playlist_manifest,
            "played_entry_ids": sorted(self._played_entry_ids),
        }
        dlg = SettingsDialog(snapshot, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.mode = dlg.settings["mode"]
        self.playlist_manifest = dlg.settings.get("playlist")
        self._played_entry_ids = set(dlg.settings.get("played_entry_ids") or [])
        self._persist_settings()
        # Settings changes end the current round so labels stay consistent.
        sd.stop()
        self.round = None
        self.game_over = False
        self._set_controls_enabled(False)
        self._apply_mode_labels()

    def _set_controls_enabled(self, enabled: bool):
        for w in (
            self.play_next_btn,
            self.replay_btn,
            self.forfeit_btn,
            self.submit_btn,
            self.guess_input,
            self.suggestions,
        ):
            w.setEnabled(enabled)
        self._update_play_next_label()

    def _rebuild_attempt_rows(self, count: int):
        while self.attempts_host.count():
            item = self.attempts_host.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.attempt_rows = []
        for i in range(count):
            row = AttemptRow(i)
            self.attempt_rows.append(row)
            self.attempts_host.addWidget(row)
        self.attempts_host.addStretch(1)

    def _filter_suggestions(self, text: str):
        needle = normalize_guess(text)
        self.suggestions.clear()
        matches = [label for label in self.labels if needle in normalize_guess(label)] if needle else self.labels
        for label in matches[:200]:
            self.suggestions.addItem(QListWidgetItem(label))

    def _pick_suggestion(self, item: QListWidgetItem):
        # Capture text before setText → textChanged → clear() deletes the item.
        guess = item.text()
        self.guess_input.blockSignals(True)
        self.guess_input.setText(guess)
        self.guess_input.blockSignals(False)
        self.submit_guess(guess)

    def _submit_from_input(self):
        text = self.guess_input.text().strip()
        if not text:
            return
        current = self.suggestions.currentItem()
        if current is not None and normalize_guess(current.text()) == normalize_guess(text):
            self.submit_guess(current.text())
            return
        for label in self.labels:
            if normalize_guess(label) == normalize_guess(text):
                self.submit_guess(label)
                return
        if self.suggestions.count() == 1:
            self.submit_guess(self.suggestions.item(0).text())
            return
        self.status_label.setText("Pick a song from the list (exact title required).")

    def start_new_game(self):
        sd.stop()

        if self.mode == "playlist":
            if not self.playlist_manifest or not self.playlist_manifest.get("entries"):
                QMessageBox.warning(
                    self,
                    "No playlist",
                    "Open Settings and load a YouTube playlist (or add tracks) first.",
                )
                return
            entry = self._pick_playlist_entry()
            if entry is None:
                QMessageBox.warning(self, "Empty playlist", "This playlist has no tracks.")
                return
            title = entry["title"]
            self._played_entry_ids.add(str(entry["id"]))
            self._persist_settings()
            self._prepare_dialog = PreparingDialog(self)
            self._worker = PrepareWorker(
                playlist_entry=entry,
                answer_label=title,
                parent=self,
            )
            self._worker.status.connect(self._prepare_dialog.set_status)
            self._worker.progress.connect(self._prepare_dialog.set_progress)
            self._worker.finished_ok.connect(self._on_prepared)
            self._worker.failed.connect(self._on_prepare_failed)
            self._worker.start()
            self._prepare_dialog.show()
            return

        self.songs = list_songs()
        self.song_by_label = {song_label(p): p for p in self.songs}
        if not self.songs:
            QMessageBox.warning(self, "No songs", f"Add audio files under:\n{SONGSDIR}")
            return

        song = random.choice(self.songs)
        self._prepare_dialog = PreparingDialog(self)
        self._worker = PrepareWorker(song_path=song, parent=self)
        self._worker.status.connect(self._prepare_dialog.set_status)
        self._worker.progress.connect(self._prepare_dialog.set_progress)
        self._worker.finished_ok.connect(self._on_prepared)
        self._worker.failed.connect(self._on_prepare_failed)
        self._worker.start()
        self._prepare_dialog.show()

    def _pick_playlist_entry(self) -> dict[str, Any] | None:
        assert self.playlist_manifest is not None
        entries = self.playlist_manifest["entries"]
        if not entries:
            return None
        remaining = [e for e in entries if str(e["id"]) not in self._played_entry_ids]
        if not remaining:
            self._played_entry_ids.clear()
            remaining = list(entries)
        return random.choice(remaining)

    def _on_prepared(self, result: dict):
        if self._prepare_dialog is not None:
            self._prepare_dialog.accept()
            self._prepare_dialog = None

        self.round = result
        self.revealed = 1
        self.guesses = []
        self.game_over = False
        self._rebuild_attempt_rows(len(result["stem_order"]))
        self.guess_input.clear()
        self._filter_suggestions("")
        self._set_controls_enabled(True)
        self.status_label.setText(
            f"Round ready · {len(result['stem_order'])} stems · "
            f"window {result['window'][0]:.1f}s–{result['window'][1]:.1f}s"
        )
        self.subtitle.setText(
            f"Guess 1 / {len(result['stem_order'])} · stems playing: {result['stem_order'][0]}"
        )
        self._rebuild_mix(True)

    def _on_prepare_failed(self, message: str):
        if self._prepare_dialog is not None:
            self._prepare_dialog.reject()
            self._prepare_dialog = None
        QMessageBox.critical(self, "Prepare failed", message)
        self.status_label.setText("Preparation failed. Try another song.")

    def active_stem_names(self) -> list[str]:
        assert self.round is not None
        return self.round["stem_order"][: self.revealed]

    def _rebuild_mix(self, autoplay: bool = True):
        if self.round is None:
            return
        with self._play_lock:
            self._current_mix = mix_stems(self.round["stems"], self.active_stem_names())
            if autoplay:
                sd.stop()
                sd.play(self._current_mix, self.round["sr"], blocking=False)

    def replay_current(self):
        if self.round is None or self._current_mix is None:
            return
        with self._play_lock:
            sd.stop()
            sd.play(self._current_mix, self.round["sr"], blocking=False)

    def forfeit_guess(self):
        """Forfeit the current guess slot, reveal next stem, and play it."""
        if self.round is None or self.game_over:
            return

        row_index = len(self.guesses)
        if row_index >= len(self.attempt_rows):
            return

        self.guesses.append("")
        self.attempt_rows[row_index].set_forfeit()
        self.guess_input.clear()

        max_guesses = len(self.round["stem_order"])
        if len(self.guesses) >= max_guesses:
            self.game_over = True
            self.revealed = max_guesses
            self._rebuild_mix(True)
            self._set_controls_enabled(False)
            self.new_btn.setEnabled(True)
            answer = self.round["label"]
            self.status_label.setText(f"Out of guesses. Answer: {answer}")
            self.subtitle.setText("Round over — click New Game.")
            return

        self.revealed = min(self.revealed + 1, max_guesses)
        active = ", ".join(self.active_stem_names())
        self.subtitle.setText(
            f"Guess {len(self.guesses) + 1} / {max_guesses} · stems playing: {active}"
        )
        self.status_label.setText(f"Forfeited guess #{row_index + 1} — next stem layered in.")
        self._update_play_next_label()
        self._rebuild_mix(True)

    def forfeit_game(self):
        if self.round is None or self.game_over:
            return
        self.game_over = True
        self.revealed = len(self.round["stem_order"])
        self._rebuild_mix(True)
        self._set_controls_enabled(False)
        self.new_btn.setEnabled(True)
        answer = self.round["label"]
        self.status_label.setText(f"Forfeited. Answer: {answer}")
        self.subtitle.setText("Round over — click New Game.")

    def submit_guess(self, guess: str):
        if self.round is None or self.game_over:
            return

        answer = self.round["label"]
        correct = normalize_guess(guess) == normalize_guess(answer)
        row_index = len(self.guesses)
        if row_index >= len(self.attempt_rows):
            return

        self.guesses.append(guess)
        self.attempt_rows[row_index].set_guess(guess, correct)
        self.guess_input.clear()

        if correct:
            self.game_over = True
            self.revealed = len(self.round["stem_order"])
            self._rebuild_mix(True)
            self._set_controls_enabled(False)
            self.new_btn.setEnabled(True)
            self.status_label.setText(f"Correct! {answer}")
            self.subtitle.setText(f"Solved in {len(self.guesses)} / {len(self.round['stem_order'])}")
            return

        max_guesses = len(self.round["stem_order"])
        if len(self.guesses) >= max_guesses:
            self.game_over = True
            self.revealed = max_guesses
            self._rebuild_mix(True)
            self._set_controls_enabled(False)
            self.new_btn.setEnabled(True)
            self.status_label.setText(f"Out of guesses. Answer: {answer}")
            self.subtitle.setText("Round over — click New Game.")
            return

        self.revealed = min(self.revealed + 1, max_guesses)
        active = ", ".join(self.active_stem_names())
        self.subtitle.setText(
            f"Guess {len(self.guesses) + 1} / {max_guesses} · stems playing: {active}"
        )
        self.status_label.setText("Wrong — next stem layered in.")
        self._update_play_next_label()
        self._rebuild_mix(True)

    def closeEvent(self, event):
        self._persist_settings()
        sd.stop()
        super().closeEvent(event)



def main():
    ensure_ffmpeg_on_path()
    SONGSDIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    window = BandleWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
