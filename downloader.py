from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import yt_dlp

StatusCb = Callable[[str], None] | None
ProgressCb = Callable[[str, float | None], None] | None


def empty_manifest() -> dict[str, Any]:
    return {
        "title": "Custom Playlist",
        "url": "",
        "entries": [],
    }


def fetch_playlist_manifest(playlist_url: str, status_cb: StatusCb = None) -> dict[str, Any]:
    if status_cb:
        status_cb("Fetching playlist manifest...")

    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch playlist: {exc}") from exc

        if not info:
            raise RuntimeError("No playlist information returned.")

        entries = []
        if "entries" in info:
            title = info.get("title") or "YouTube Playlist"
            for item in info["entries"]:
                if item:
                    entries.append({
                        "id": item.get("id") or item.get("url"),
                        "title": item.get("title") or item.get("id") or "Untitled",
                        "url": item.get("url") or f"https://www.youtube.com/watch?v={item.get('id')}",
                    })
        else:
            title = info.get("title") or "Single Video"
            entries.append({
                "id": info.get("id"),
                "title": title,
                "url": playlist_url,
            })

        return {
            "title": title,
            "url": playlist_url,
            "entries": entries,
        }


def fetch_video_entry(url: str, status_cb: StatusCb = None) -> dict[str, Any]:
    if status_cb:
        status_cb("Fetching video entry info...")

    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch video entry: {exc}") from exc

        if not info:
            raise RuntimeError("No video information returned.")

        return {
            "id": info.get("id") or url,
            "title": info.get("title") or info.get("id") or "Untitled",
            "url": url,
        }


def download_track(entry: dict[str, Any], yt_download_dir: Path, progress_cb: ProgressCb = None) -> Path:
    yt_download_dir.mkdir(parents=True, exist_ok=True)
    video_id = entry.get("id")
    video_url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
    title = entry.get("title", "Unknown Track")

    if progress_cb:
        progress_cb(f"Downloading {title}...", 0.0)

    # Temporary download tracking
    def ydl_hook(d: dict[str, Any]) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                frac = downloaded / total
                if progress_cb:
                    progress_cb(f"Downloading {title} ({int(frac * 100)}%)...", frac)
            else:
                if progress_cb:
                    progress_cb(f"Downloading {title}...", None)
        elif d.get("status") == "finished":
            if progress_cb:
                progress_cb("Download complete, extracting audio...", 1.0)

    # Let's write downloaded file as video_id.mp3
    output_template = str(yt_download_dir / f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "progress_hooks": [ydl_hook],
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    # Remove existing mp3 file if it already exists to ensure fresh download
    final_path = yt_download_dir / f"{video_id}.mp3"
    if final_path.exists():
        try:
            final_path.unlink()
        except OSError:
            pass

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
        except Exception as exc:
            raise RuntimeError(f"Failed to download audio: {exc}") from exc

    if not final_path.exists():
        # Fallback to check if a file was downloaded but with some other ext/name
        # though FFmpegExtractAudio should produce .mp3
        downloaded_files = list(yt_download_dir.glob(f"{video_id}.*"))
        if downloaded_files:
            return downloaded_files[0]
        raise RuntimeError(f"Expected audio file not found: {final_path}")

    return final_path
