from __future__ import annotations

import sys
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from runtime_env import ensure_runtime, is_runtime_ready, launch_app


class SetupWorker(QThread):
    status_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(float)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            ensure_runtime(
                status_cb=self.status_changed.emit,
                progress_cb=self.progress_changed.emit,
            )
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class SetupWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bandle - Standalone Setup")
        self.resize(500, 180)

        # Dark modern GitHub-like style matching main app
        self.setStyleSheet(
            """
            QWidget {
                background: #0d1117;
                color: #e6edf3;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                font-size: 13px;
            }
            QLabel {
                color: #8b949e;
            }
            QLabel#title {
                color: #58a6ff;
                font-size: 18px;
                font-weight: bold;
            }
            QProgressBar {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                text-align: center;
                height: 22px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: #1f6feb;
                border-radius: 5px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title_label = QLabel("Welcome to Bandle-local")
        title_label.setObjectName("title")
        layout.addWidget(title_label)

        self.info_label = QLabel("Configuring standalone audio processing runtime (first-time setup)...")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Start worker
        self.worker = SetupWorker(self)
        self.worker.status_changed.connect(self._on_status)
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_progress(self, frac: float) -> None:
        val = int(frac * 100)
        self.progress_bar.setValue(val)

    def _on_finished(self) -> None:
        self.close()

    def _on_failed(self, error_msg: str) -> None:
        QMessageBox.critical(
            self,
            "Setup Error",
            f"An error occurred during setup:\n\n{error_msg}\n\nPlease try running the application again.",
        )
        sys.exit(1)


def main() -> int:
    # If already set up, skip GUI entirely and run the app immediately
    if is_runtime_ready():
        return launch_app()

    app = QApplication(sys.argv)
    window = SetupWindow()
    window.show()
    app.exec()

    # Once window closes (setup done or failed)
    if is_runtime_ready():
        return launch_app()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
