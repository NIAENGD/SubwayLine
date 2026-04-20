"""CLI and GUI entrypoint for SubwayLine."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from config.defaults import build_default_config


def _load_config_io_module():
    module_path = Path(__file__).resolve().parent / "io" / "config_io.py"
    spec = importlib.util.spec_from_file_location("subwayline_config_io", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config I/O module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SubwayLine entrypoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--save-config", type=str, default=None, help="Write resolved config JSON")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run configuration load/save smoke path and exit without launching the GUI",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace):
    config_io = _load_config_io_module()
    config = config_io.load_or_default(args.config) if args.config else build_default_config()

    if args.save_config:
        config_io.save_config(config, args.save_config)

    return config


def _run_gui(config) -> int:
    from PySide6.QtWidgets import QApplication

    from gui.main_window import MainWindow

    app = QApplication([])
    window = MainWindow(initial_config=config)
    window.show()
    return app.exec()


def main() -> int:
    args = parse_args()
    config = _resolve_config(args)

    print("SubwayLine initialized")
    print(f"schema_version={config.schema_version}")
    print(f"seed_context={config.seed_context}")

    if args.no_gui:
        return 0

    try:
        return _run_gui(config)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(
            "Unable to launch GUI because a required dependency is missing: "
            f"{missing}. Install GUI/runtime deps (for example: numpy, matplotlib, PySide6)."
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
