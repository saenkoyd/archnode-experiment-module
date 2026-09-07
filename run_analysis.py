#!/usr/bin/env python3
"""Unified launcher for the architecture-method comparison module.

The launcher executes the core structural-parametric analysis and, unless
disabled, the additional TOPSIS/VIKOR comparison.  Only the Python standard
library is required.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Модуль анализа и сравнения методов структурно-параметрического "
            "синтеза архитектуры сложных программных систем"
        )
    )
    parser.add_argument(
        "--without-topsis-vikor",
        action="store_true",
        help="выполнить только основной анализ без дополнительного сравнения",
    )
    return parser.parse_args()


def run(script_name: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / script_name)],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    print("Модуль анализа и сравнения методов, версия 1.0", flush=True)
    print("Выполняется основной структурно-параметрический анализ...", flush=True)
    run("run_experiment.py")
    if not args.without_topsis_vikor:
        print("Выполняется дополнительное сравнение TOPSIS/VIKOR...", flush=True)
        run("run_comparative_methods.py")
    print(
        "Анализ завершён. Результаты сохранены в experiment/results.",
        flush=True,
    )


if __name__ == "__main__":
    main()
