from __future__ import annotations

import sys

from runtime_env import ensure_runtime, launch_app


def _status(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ensure_runtime(status_cb=_status)
    return launch_app()


if __name__ == "__main__":
    raise SystemExit(main())
