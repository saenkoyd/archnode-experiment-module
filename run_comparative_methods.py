#!/usr/bin/env python3
"""Reproducible TOPSIS/VIKOR comparison for dissertation chapter 4.

The script uses the same configuration, generated alternatives, feasibility
rules, criterion weights, holdout workloads, and reference full-method choices
as ``run_experiment.py``.  It adds no expert data and uses only the Python
standard library.

Run from the repository root:

    python3 experiment/run_comparative_methods.py

By default, five files are written to ``experiment/results``.  Use
``--output-dir`` to verify the calculation without changing that directory.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import run_experiment as core


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results"
METHODS = ("TOPSIS", "VIKOR")
REGIMES = ("no", "medium", "strong", "conflict")


@dataclass(frozen=True)
class Candidate:
    state: tuple[str, ...]
    vector: tuple[float, float, float]
    utility: float
    p95_latency_ms: float
    error_rate: float
    recovery_s: float
    resource_index: float


@dataclass(frozen=True)
class RankingResult:
    candidate: Candidate
    score: float
    direction: str
    vikor_s: float | None = None
    vikor_r: float | None = None


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def system_vector(state: Sequence[core.Alternative]) -> tuple[float, float, float]:
    return tuple(
        statistics.fmean(alternative.f[index] for alternative in state)
        for index in range(3)
    )  # type: ignore[return-value]


def select_topsis(
    candidates: Sequence[Candidate], weights: Sequence[float]
) -> RankingResult:
    """Select the maximum TOPSIS closeness; all criteria are benefit criteria."""
    if not candidates:
        raise ValueError("TOPSIS requires at least one candidate")
    dimensions = len(weights)
    denominators = [
        math.sqrt(sum(candidate.vector[k] ** 2 for candidate in candidates))
        for k in range(dimensions)
    ]
    weighted_rows = [
        tuple(
            weights[k]
            * (candidate.vector[k] / denominators[k] if denominators[k] else 0.0)
            for k in range(dimensions)
        )
        for candidate in candidates
    ]
    positive = [max(row[k] for row in weighted_rows) for k in range(dimensions)]
    negative = [min(row[k] for row in weighted_rows) for k in range(dimensions)]
    ranked: list[tuple[float, Candidate]] = []
    for candidate, row in zip(candidates, weighted_rows):
        distance_positive = math.sqrt(
            sum((row[k] - positive[k]) ** 2 for k in range(dimensions))
        )
        distance_negative = math.sqrt(
            sum((row[k] - negative[k]) ** 2 for k in range(dimensions))
        )
        denominator = distance_positive + distance_negative
        closeness = distance_negative / denominator if denominator else 1.0
        ranked.append((closeness, candidate))
    score, selected = max(
        ranked,
        key=lambda item: (item[0], tuple(reversed(item[1].state))),
    )
    return RankingResult(selected, score, "max")


def select_vikor(
    candidates: Sequence[Candidate],
    weights: Sequence[float],
    compromise_weight: float = 0.5,
) -> RankingResult:
    """Select the minimum VIKOR index without expanding to a compromise set."""
    if not candidates:
        raise ValueError("VIKOR requires at least one candidate")
    if not 0.0 <= compromise_weight <= 1.0:
        raise ValueError("VIKOR compromise weight must be in [0, 1]")
    dimensions = len(weights)
    best = [max(candidate.vector[k] for candidate in candidates) for k in range(dimensions)]
    worst = [min(candidate.vector[k] for candidate in candidates) for k in range(dimensions)]
    components: list[tuple[float, float, Candidate]] = []
    for candidate in candidates:
        weighted_gaps = [
            weights[k] * (best[k] - candidate.vector[k]) / (best[k] - worst[k])
            if best[k] - worst[k] > 1e-15
            else 0.0
            for k in range(dimensions)
        ]
        components.append((sum(weighted_gaps), max(weighted_gaps), candidate))

    s_best = min(item[0] for item in components)
    s_worst = max(item[0] for item in components)
    r_best = min(item[1] for item in components)
    r_worst = max(item[1] for item in components)
    ranked: list[tuple[float, float, float, Candidate]] = []
    for s_value, r_value, candidate in components:
        normalized_s = (
            (s_value - s_best) / (s_worst - s_best)
            if s_worst - s_best > 1e-15
            else 0.0
        )
        normalized_r = (
            (r_value - r_best) / (r_worst - r_best)
            if r_worst - r_best > 1e-15
            else 0.0
        )
        index = (
            compromise_weight * normalized_s
            + (1.0 - compromise_weight) * normalized_r
        )
        ranked.append((index, s_value, r_value, candidate))
    score, s_value, r_value, selected = min(
        ranked, key=lambda item: (item[0], item[3].state)
    )
    return RankingResult(selected, score, "min", s_value, r_value)


def rank(
    method: str, candidates: Sequence[Candidate], weights: Sequence[float]
) -> RankingResult:
    if method == "TOPSIS":
        return select_topsis(candidates, weights)
    if method == "VIKOR":
        return select_vikor(candidates, weights)
    raise ValueError(f"unknown method: {method}")


def control_comparison(weights: Sequence[float]) -> list[dict[str, object]]:
    alternatives = {
        row["alternative"]: tuple(float(row[f"f{k}"]) for k in range(1, 4))
        for row in load_csv(ROOT / "results" / "control_alternatives_calculated.csv")
    }
    candidates: list[Candidate] = []
    for row in load_csv(ROOT / "results" / "control_states_raw.csv"):
        names = (row["A"], row["B"], row["C"])
        vector = tuple(
            statistics.fmean(alternatives[name][k] for name in names) for k in range(3)
        )
        candidates.append(
            Candidate(
                state=(row["state"],),
                vector=vector,  # type: ignore[arg-type]
                utility=float(row["control_utility"]),
                p95_latency_ms=float(row["p95_latency_ms"]),
                error_rate=float(row["error_rate"]),
                recovery_s=float(row["recovery_s"]),
                resource_index=float(row["resource_index"]),
            )
        )
    best_utility = max(candidate.utility for candidate in candidates)
    worst_utility = min(candidate.utility for candidate in candidates)
    reference = load_json(ROOT / "inputs" / "control_expectations.json")[
        "full_method_state"
    ]
    rows: list[dict[str, object]] = []
    for method in METHODS:
        result = rank(method, candidates, weights)
        rows.append(
            {
                "method": method,
                "selected_state": result.candidate.state[0],
                "ranking_score": f"{result.score:.12f}",
                "ranking_direction": result.direction,
                "vikor_s": (
                    f"{result.vikor_s:.12f}" if result.vikor_s is not None else ""
                ),
                "vikor_r": (
                    f"{result.vikor_r:.12f}" if result.vikor_r is not None else ""
                ),
                "reference_state": reference,
                "top1_agreement": int(result.candidate.state[0] == reference),
                "control_utility": f"{result.candidate.utility:.12f}",
                "normalized_regret": f"{core.normalized_regret(result.candidate.utility, best_utility, worst_utility):.12f}",
                "p95_latency_ms": f"{result.candidate.p95_latency_ms:.12f}",
                "error_rate": f"{result.candidate.error_rate:.12f}",
                "recovery_s": f"{result.candidate.recovery_s:.12f}",
                "resource_index": f"{result.candidate.resource_index:.12f}",
            }
        )
    assert rows[0]["selected_state"] == "S1"
    assert rows[1]["selected_state"] == "S3"
    return rows


def load_full_method_rows() -> dict[tuple[str, int], dict[str, str]]:
    lookup = {
        (row["regime"], int(row["replicate"])): row
        for row in load_csv(ROOT / "results" / "batch_raw.csv")
        if row["method"] == "full-method"
    }
    if len(lookup) != 120:
        raise AssertionError("batch_raw.csv must contain 120 full-method rows")
    return lookup


def batch_comparison(config: dict) -> list[dict[str, object]]:
    batch_config = config["batch"]
    weights = tuple(config["criterion_weights"])
    full_rows = load_full_method_rows()
    master = random.Random(config["master_seed"])
    task_seeds = [
        master.randrange(1, 2**31)
        for _ in range(batch_config["replicates_per_regime"])
    ]
    rows: list[dict[str, object]] = []
    for regime in REGIMES:
        regime_config = batch_config["regimes"][regime]
        for replicate, seed in enumerate(task_seeds, 1):
            alternatives, metadata = core.generate_batch_task(
                regime, replicate, seed, batch_config, weights
            )
            levels = sorted(alternatives)
            holdout_rng = random.Random(int(metadata["holdout_seed"]))
            workloads = tuple(
                0.25 + 0.70 * holdout_rng.random()
                for _ in range(int(metadata["holdout_samples"]))
            )
            states = [
                state
                for state in itertools.product(
                    *(alternatives[level] for level in levels)
                )
                if core.batch_state_feasible(
                    state,
                    regime,
                    seed,
                    regime_config["conflict_probability"],
                )
            ]
            candidates: list[Candidate] = []
            for state in states:
                utility, latency, error_rate, recovery, resource = (
                    core.batch_control_outcomes(state, workloads)
                )
                candidates.append(
                    Candidate(
                        state=tuple(alternative.name for alternative in state),
                        vector=system_vector(state),
                        utility=utility,
                        p95_latency_ms=latency,
                        error_rate=error_rate,
                        recovery_s=recovery,
                        resource_index=resource,
                    )
                )
            best_utility = max(candidate.utility for candidate in candidates)
            worst_utility = min(candidate.utility for candidate in candidates)
            full = full_rows[(regime, replicate)]
            if int(full["seed"]) != seed:
                raise AssertionError(f"seed mismatch for {regime}/{replicate}")
            if abs(float(full["holdout_best_utility"]) - best_utility) > 1e-10:
                raise AssertionError(f"holdout mismatch for {regime}/{replicate}")
            reference_state = tuple(full["selected_state"].split("|"))
            reference_regret = float(full["normalized_regret"])
            for method in METHODS:
                result = rank(method, candidates, weights)
                selected = result.candidate
                regret = core.normalized_regret(
                    selected.utility, best_utility, worst_utility
                )
                if regret is None:
                    raise AssertionError("a feasible comparative choice must have regret")
                rows.append(
                    {
                        "regime": regime,
                        "replicate": replicate,
                        "seed": seed,
                        "method": method,
                        "selected_state": "|".join(selected.state),
                        "ranking_score": f"{result.score:.12f}",
                        "ranking_direction": result.direction,
                        "vikor_s": (
                            f"{result.vikor_s:.12f}"
                            if result.vikor_s is not None
                            else ""
                        ),
                        "vikor_r": (
                            f"{result.vikor_r:.12f}"
                            if result.vikor_r is not None
                            else ""
                        ),
                        "g1": f"{selected.vector[0]:.12f}",
                        "g2": f"{selected.vector[1]:.12f}",
                        "g3": f"{selected.vector[2]:.12f}",
                        "feasible_states": len(candidates),
                        "reference_state": "|".join(reference_state),
                        "top1_agreement": int(selected.state == reference_state),
                        "control_utility": f"{selected.utility:.12f}",
                        "normalized_regret": f"{regret:.12f}",
                        "reference_regret": f"{reference_regret:.12f}",
                        "regret_difference": f"{regret - reference_regret:.12f}",
                        "p95_latency_ms": f"{selected.p95_latency_ms:.12f}",
                        "error_rate": f"{selected.error_rate:.12f}",
                        "recovery_s": f"{selected.recovery_s:.12f}",
                        "resource_index": f"{selected.resource_index:.12f}",
                    }
                )
    if len(rows) != 240:
        raise AssertionError(f"expected 240 comparative rows, got {len(rows)}")
    return rows


def summarize(
    rows: Sequence[dict[str, object]], master_seed: int, resamples: int
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for regime_index, regime in enumerate(REGIMES):
        for method_index, method in enumerate(METHODS):
            group = [
                row
                for row in rows
                if row["regime"] == regime and row["method"] == method
            ]
            regrets = [float(row["normalized_regret"]) for row in group]
            differences = [float(row["regret_difference"]) for row in group]
            q1 = core.percentile(regrets, 0.25)
            q3 = core.percentile(regrets, 0.75)
            low, high = core.bootstrap_mean_ci(
                differences,
                master_seed + 5000 + regime_index * 100 + method_index,
                resamples,
            )
            summary.append(
                {
                    "regime": regime,
                    "method": method,
                    "n": len(group),
                    "top1_agreement_rate": f"{statistics.fmean(int(row['top1_agreement']) for row in group):.6f}",
                    "mean_normalized_regret": f"{statistics.fmean(regrets):.6f}",
                    "median_normalized_regret": f"{statistics.median(regrets):.6f}",
                    "q1_normalized_regret": f"{q1:.6f}",
                    "q3_normalized_regret": f"{q3:.6f}",
                    "iqr_normalized_regret": f"{q3 - q1:.6f}",
                    "mean_regret_difference": f"{statistics.fmean(differences):.6f}",
                    "bootstrap_ci95_low": f"{low:.6f}",
                    "bootstrap_ci95_high": f"{high:.6f}",
                    "full_better_count": sum(value > 1e-12 for value in differences),
                    "tie_count": sum(abs(value) <= 1e-12 for value in differences),
                    "full_worse_count": sum(value < -1e-12 for value in differences),
                    "mean_p95_latency_ms": f"{statistics.fmean(float(row['p95_latency_ms']) for row in group):.6f}",
                    "mean_error_rate": f"{statistics.fmean(float(row['error_rate']) for row in group):.6f}",
                    "mean_recovery_s": f"{statistics.fmean(float(row['recovery_s']) for row in group):.6f}",
                    "mean_resource_index": f"{statistics.fmean(float(row['resource_index']) for row in group):.6f}",
                }
            )
    return summary


EXPECTED = {
    ("no", "TOPSIS"): ("0.500000", "0.233174", "0.043551", "-0.018152", "0.123640"),
    ("no", "VIKOR"): ("0.366667", "0.226336", "0.036713", "-0.011298", "0.087375"),
    ("medium", "TOPSIS"): ("0.600000", "0.233174", "0.066487", "-0.003457", "0.150999"),
    ("medium", "VIKOR"): ("0.466667", "0.226336", "0.059650", "0.019824", "0.108138"),
    ("strong", "TOPSIS"): ("0.433333", "0.233174", "0.125157", "0.043580", "0.219937"),
    ("strong", "VIKOR"): ("0.366667", "0.226336", "0.118320", "0.063592", "0.176792"),
    ("conflict", "TOPSIS"): ("0.666667", "0.179304", "0.083606", "0.033132", "0.140539"),
    ("conflict", "VIKOR"): ("0.600000", "0.200374", "0.104676", "0.046887", "0.173033"),
}


def assert_expected(summary: Sequence[dict[str, object]], resamples: int) -> None:
    if resamples != 5000:
        return
    for row in summary:
        key = (str(row["regime"]), str(row["method"]))
        actual = (
            row["top1_agreement_rate"],
            row["mean_normalized_regret"],
            row["mean_regret_difference"],
            row["bootstrap_ci95_low"],
            row["bootstrap_ci95_high"],
        )
        if actual != EXPECTED[key]:
            raise AssertionError(f"summary mismatch for {key}: {actual} != {EXPECTED[key]}")


def markdown_tables(
    control_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# Сравнение с TOPSIS и VIKOR",
        "",
        "## Контрольный пример",
        "",
        "| Метод | Выбор | Показатель ранжирования | Внецелевая полезность |",
        "|---|---|---:|---:|",
    ]
    for row in control_rows:
        lines.append(
            f"| {row['method']} | {row['selected_state']} | {row['ranking_score']} | {row['control_utility']} |"
        )
    lines.extend(
        [
            "",
            "## Серийный эксперимент",
            "",
            "| Режим | Метод | Совпадение top-1 | Средняя потеря | Парная разность | 95 %-й интервал |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['regime']} | {row['method']} | {row['top1_agreement_rate']} | "
            f"{row['mean_normalized_regret']} | {row['mean_regret_difference']} | "
            f"[{row['bootstrap_ci95_low']}; {row['bootstrap_ci95_high']}] |"
        )
    lines.extend(
        [
            "",
            "Парная разность определена как потеря сравниваемого метода минус потеря предложенного процесса. Положительное значение означает меньшую потерю у предложенного процесса.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for comparative CSV, JSON, and Markdown outputs",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=5000,
        help="number of paired bootstrap resamples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    config = load_json(ROOT / "config.json")
    control_rows = control_comparison(tuple(config["criterion_weights"]))
    raw_rows = batch_comparison(config)
    summary_rows = summarize(
        raw_rows, int(config["master_seed"]), args.bootstrap_resamples
    )
    assert_expected(summary_rows, args.bootstrap_resamples)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "comparative_control.csv", control_rows)
    write_csv(output_dir / "comparative_methods_raw.csv", raw_rows)
    write_csv(output_dir / "comparative_methods_summary.csv", summary_rows)
    payload = {
        "schema_version": 1,
        "master_seed": config["master_seed"],
        "methods": list(METHODS),
        "control": control_rows,
        "summary": summary_rows,
        "assertions": {
            "same_generated_tasks_and_holdout_workloads": "passed",
            "control_topsis_selects_S1": "passed",
            "control_vikor_selects_S3": "passed",
            "all_240_comparative_results_present": "passed",
            "manuscript_summary_values_match": "passed",
        },
    }
    (output_dir / "comparative_methods_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "comparative_methods_tables.md").write_text(
        markdown_tables(control_rows, summary_rows), encoding="utf-8"
    )
    print(f"wrote comparative results to {output_dir}")
    print("all TOPSIS/VIKOR reproducibility assertions passed")


if __name__ == "__main__":
    main()
