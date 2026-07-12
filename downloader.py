"""YouTube playlist helpers: flat manifest fetch + single-track download."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import yt_dlp

StatusCb = Callable[[str], None] | None
# fraction is 0.0–1.0 when known, or None for indeterminate.
ProgressCb = Callable[[str, float | None], None] | None


def _safe_filename(name: str, max_len: int = 120) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return (cleaned or "track")[:max_len]


def entry_url(entry: dict[str, Any]) -> str:
    url = entry.get("url") or entry.get("webpage_url")
    if url and url.startswith("http"):
        return url
    video_id = entry.get("id")
    if not video_id:
        raise ValueError("Playlist entry has no URL or id")
    return f"https://www.youtube.com/watch?v={video_id}"


def entry_title(entry: dict[str, Any]) -> str:
    return (entry.get("title") or entry.get("id") or "Unknown Title").strip()


def _normalize_entry(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        url = entry_url(raw)
    except ValueError:
        return None
    return {
        "id": str(raw.get("id") or url),
        "title": entry_title(raw),
        "url": url,
    }


def empty_manifest(title: str = "Custom list") -> dict[str, Any]:
    return {"url": "", "title": title, "entries": []}


def fetch_playlist_manifest(playlist_url: str, status_cb: StatusCb = None) -> dict[str, Any]:
    """
    Flat-extract a playlist (metadata only, no media download).

    Returns {"url", "title", "entries": [{"id", "title", "url"}, ...]}.
    """
    if status_cb:
        status_cb("Fetching playlist manifest…")

    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    if not info or not info.get("entries"):
        raise RuntimeError(
            "Could not find any entries in this playlist. Ensure it is public or unlisted."
        )

    entries: list[dict[str, Any]] = []
    for raw in info["entries"]:
        entry = _normalize_entry(raw) if isinstance(raw, dict) else None
        if entry:
            entries.append(entry)

    if not entries:
        raise RuntimeError("Playlist had no usable video entries.")

    return {
        "url": playlist_url,
        "title": info.get("title") or "YouTube playlist",
        "entries": entries,
    }


def fetch_video_entry(video_url: str, status_cb: StatusCb = None) -> dict[str, Any]:
    """Resolve a single YouTube (or yt-dlp) URL into a manifest entry."""
    if status_cb:
        status_cb("Resolving YouTube link…")

    ydl_opts = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    if not info:
        raise RuntimeError("Could not resolve that YouTube link.")

    # If someone pasted a playlist URL into the add-track field, take first entry.
    if info.get("entries"):
        for raw in info["entries"]:
            entry = _normalize_entry(raw) if isinstance(raw, dict) else None
            if entry:
                return entry
        raise RuntimeError("Playlist link had no usable videos. Load it via Mode instead.")

    entry = _normalize_entry(info)
    if not entry:
        raise RuntimeError("Could not resolve that YouTube link.")
    return entry


def download_track(
    entry: dict[str, Any],
    dest_dir: Path,
    status_cb: StatusCb = None,
    progress_cb: ProgressCb = None,
) -> Path:
    """Download one playlist entry as high-quality MP3 into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    title = entry_title(entry)
    video_id = str(entry.get("id") or "track")
    out_stem = f"{_safe_filename(title)} [{video_id}]"
    out_path = dest_dir / f"{out_stem}.mp3"

    def emit(msg: str, frac: float | None = None):
        if progress_cb:
            progress_cb(msg, frac)
        elif status_cb:
            status_cb(msg)

    if out_path.is_file() and out_path.stat().st_size > 0:
        emit("Using cached download", 1.0)
        return out_path

    # Clear stale partials with the same stem.
    for stale in dest_dir.glob(f"{out_stem}.*"):
        if stale.suffix.lower() != ".mp3":
            try:
                stale.unlink()
            except OSError:
                pass

    emit("Downloading…", 0.0)

    def hook(d: dict[str, Any]):
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or 0
            if total:
                pct = min(done / total, 0.99)
                emit(f"Downloading… {int(pct * 100)}%", pct)
            else:
                emit("Downloading…", None)
        elif status == "finished":
            emit("Converting audio…", 0.99)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / f"{out_stem}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "0",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([entry_url(entry)])

    if not out_path.is_file():
        # Fallback: pick newest mp3 written for this stem.
        candidates = sorted(
            dest_dir.glob(f"{out_stem}*.mp3"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError("Download finished but MP3 not found.")
        out_path = candidates[0]

    emit("Download complete", 1.0)
    return out_path
