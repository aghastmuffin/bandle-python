"""Runtime bootstrap paths and installer (stdlib only — safe for thin launcher)."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Callable

# Bump when runtime-requirements.txt or install steps change.
RUNTIME_VERSION = "1"

StatusCb = Callable[[str], None] | None
ProgressCb = Callable[[float | None], None] | None


def project_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def support_dir() -> Path:
    """Writable per-user app data (outside the onefile bundle)."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Bandle"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Bandle"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "Bandle"
    base.mkdir(parents=True, exist_ok=True)
    return base


def data_dir() -> Path:
    """
    Songs, settings, downloads.
    - Frozen / launcher: Application Support
    - Dev (`python app.py`): project root
    """
    override = os.environ.get("BANDLE_DATA_DIR")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        return support_dir()
    return project_dir()


def runtime_dir() -> Path:
    return support_dir() / "runtime"


def runtime_python() -> Path:
    if sys.platform == "win32":
        return runtime_dir() / "Scripts" / "python.exe"
    return runtime_dir() / "bin" / "python"


def runtime_ready_marker() -> Path:
    return runtime_dir() / ".ready"


def is_runtime_ready() -> bool:
    marker = runtime_ready_marker()
    if not marker.is_file():
        return False
    if not runtime_python().is_file():
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == RUNTIME_VERSION
    except OSError:
        return False


def requirements_file() -> Path:
    return project_dir() / "runtime-requirements.txt"


def _pip() -> Path:
    if sys.platform == "win32":
        return runtime_dir() / "Scripts" / "pip.exe"
    return runtime_dir() / "bin" / "pip"


def _run(cmd: list[str], status_cb: StatusCb = None) -> None:
    if status_cb:
        status_cb(" ".join(cmd[:3]) + ("…" if len(cmd) > 3 else ""))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line and status_cb:
            # Keep UI readable — show last useful fragment.
            status_cb(line[-120:])
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")


def create_runtime_venv(status_cb: StatusCb = None) -> None:
    rd = runtime_dir()
    if status_cb:
        status_cb("Creating runtime environment…")
    if rd.exists() and not is_runtime_ready():
        # Incomplete/outdated — recreate.
        import shutil

        shutil.rmtree(rd, ignore_errors=True)
    rd.parent.mkdir(parents=True, exist_ok=True)
    venv.create(rd, with_pip=True, upgrade_deps=True)


def install_runtime_packages(status_cb: StatusCb = None, progress_cb: ProgressCb = None) -> None:
    req = requirements_file()
    if not req.is_file():
        raise RuntimeError(f"Missing requirements file: {req}")
    if progress_cb:
        progress_cb(None)
    if status_cb:
        status_cb("Installing packages (this is the big download)…")
    # Upgrade pip first for wheel compatibility.
    _run([str(runtime_python()), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], status_cb)
    if progress_cb:
        progress_cb(0.15)
    _run(
        [
            str(runtime_python()),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "-r",
            str(req),
        ],
        status_cb,
    )
    if progress_cb:
        progress_cb(0.85)


def prefetch_model(status_cb: StatusCb = None, progress_cb: ProgressCb = None) -> None:
    if status_cb:
        status_cb("Downloading Demucs model weights…")
    if progress_cb:
        progress_cb(0.9)
    code = (
        "from demucs.pretrained import get_model\n"
        "get_model('htdemucs_6s')\n"
        "print('model ready')\n"
    )
    _run([str(runtime_python()), "-c", code], status_cb)
    if progress_cb:
        progress_cb(0.97)


def ensure_ffmpeg_on_path() -> None:
    """Prefer bundled imageio-ffmpeg binary when present."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = str(Path(ffmpeg).parent)
        path = os.environ.get("PATH", "")
        if bin_dir not in path.split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + path
    except Exception:
        pass


def mark_runtime_ready() -> None:
    runtime_ready_marker().write_text(RUNTIME_VERSION + "\n", encoding="utf-8")


def ensure_runtime(status_cb: StatusCb = None, progress_cb: ProgressCb = None) -> None:
    if is_runtime_ready():
        if status_cb:
            status_cb("Runtime ready.")
        if progress_cb:
            progress_cb(1.0)
        return
    create_runtime_venv(status_cb)
    if progress_cb:
        progress_cb(0.05)
    install_runtime_packages(status_cb, progress_cb)
    prefetch_model(status_cb, progress_cb)
    mark_runtime_ready()
    if status_cb:
        status_cb("Runtime installed.")
    if progress_cb:
        progress_cb(1.0)


def launch_app() -> int:
    """Run the PyQt game under the downloaded runtime interpreter."""
    ensure_ffmpeg_on_path()
    data = support_dir() if getattr(sys, "frozen", False) else data_dir()
    (data / "songs").mkdir(parents=True, exist_ok=True)
    (data / "defaults").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["BANDLE_DATA_DIR"] = str(data)
    # App modules live next to launcher / in the onefile extract dir.
    code_dir = str(project_dir())
    env["PYTHONPATH"] = code_dir + os.pathsep + env.get("PYTHONPATH", "")

    py = runtime_python()
    if not py.is_file():
        raise RuntimeError(f"Runtime Python missing: {py}")

    # Prefer -m style once packaged; flat modules work via PYTHONPATH.
    cmd = [str(py), "-c", "from app import main; main()"]
    return subprocess.call(cmd, env=env)
