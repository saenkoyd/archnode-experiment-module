#!/usr/bin/env python3
"""Reproducible synthetic experiment for dissertation draft revision 4.

Only the Python standard library is used.  All stochastic input is generated from
the master seed in config.json.  The same generated task is passed to every
compared procedure.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
METHODS = (
    "local-greedy",
    "feasible-additive-no-trace",
    "unsafe-pareto",
    "full-method",
    "oracle-full-model",
)


@dataclass(frozen=True)
class Alternative:
    level: str
    name: str
    f: tuple[float, float, float]
    q0: float
    signature: str
    interface_fit: float
    latent: tuple[float, float, float]
    dominated_by: str | None = None
    safe_certificate: bool = False


@dataclass(frozen=True)
class StateEvaluation:
    state: tuple[str, ...]
    q0: float
    q: float
    vector: tuple[float, ...]
    iterations: int
    residual: float
    control_utility: float
    trace_bias: float
    p95_latency_ms: float
    error_rate: float
    recovery_s: float
    resource_index: float


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def weighted(values: Sequence[float], weights: Sequence[float]) -> float:
    assert len(values) == len(weights)
    return sum(value * weight for value, weight in zip(values, weights))


def dominates(left: Sequence[float], right: Sequence[float], tol: float = 1e-12) -> bool:
    return all(a >= b - tol for a, b in zip(left, right)) and any(
        a > b + tol for a, b in zip(left, right)
    )


def pareto_front(evaluations: Sequence[StateEvaluation]) -> list[StateEvaluation]:
    result: list[StateEvaluation] = []
    for candidate in evaluations:
        if not any(
            other.state != candidate.state and dominates(other.vector, candidate.vector)
            for other in evaluations
        ):
            result.append(candidate)
    return sorted(result, key=lambda item: item.state)


def local_front(alternatives: Sequence[Alternative]) -> list[Alternative]:
    return sorted(
        [
            candidate
            for candidate in alternatives
            if not any(
                other.name != candidate.name and dominates(other.f, candidate.f)
                for other in alternatives
            )
        ],
        key=lambda item: item.name,
    )


def transfer_set(alternatives: Sequence[Alternative]) -> list[Alternative]:
    """Remove a dominated alternative only when its safe certificate is present."""
    result: list[Alternative] = []
    by_name = {item.name: item for item in alternatives}
    for item in alternatives:
        removable = False
        if item.safe_certificate and item.dominated_by:
            dominator = by_name[item.dominated_by]
            removable = dominates(dominator.f, item.f)
            assert dominator.signature == item.signature
            assert abs(dominator.interface_fit - item.interface_fit) <= 1e-12
        if not removable:
            result.append(item)
    return sorted(result, key=lambda item: item.name)


def trace_adjust(
    base_vectors: Sequence[Sequence[float]],
    rho: float,
    bias: float,
    epsilon: float,
    max_iterations: int,
) -> tuple[tuple[tuple[float, ...], ...], int, float, tuple[tuple[int, float], ...]]:
    """Synchronous contractive interlevel update.

    y[i,k](t+1) = (1-rho) F[i,k] + rho * mean(j != i, y[j,k](t) + bias).
    Its sup-norm Lipschitz constant is no greater than rho, hence rho < 1 is a
    sufficient contraction condition for this experimental operator.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must satisfy 0 <= rho < 1")
    base = tuple(tuple(float(value) for value in row) for row in base_vectors)
    if rho == 0.0:
        return base, 0, 0.0, tuple()
    current = base
    residual = math.inf
    history: list[tuple[int, float]] = []
    for iteration in range(1, max_iterations + 1):
        next_rows: list[tuple[float, ...]] = []
        for i, row in enumerate(base):
            next_row: list[float] = []
            for k, base_value in enumerate(row):
                predecessors = [current[j][k] for j in range(len(base)) if j != i]
                transmitted = clip(statistics.fmean(predecessors) + bias)
                next_row.append((1.0 - rho) * base_value + rho * transmitted)
            next_rows.append(tuple(next_row))
        updated = tuple(next_rows)
        residual = max(
            abs(updated[i][k] - current[i][k])
            for i in range(len(updated))
            for k in range(len(updated[i]))
        )
        history.append((iteration, residual))
        current = updated
        if residual <= epsilon:
            return current, iteration, residual, tuple(history)
    raise AssertionError(
        f"trace adjustment did not converge: rho={rho}, residual={residual}, "
        f"max_iterations={max_iterations}"
    )


def aggregate_scores(
    vectors: Sequence[Sequence[float]],
    criterion_weights: Sequence[float],
    level_weights: Sequence[float] | None = None,
) -> tuple[list[float], float]:
    levels = [weighted(row, criterion_weights) for row in vectors]
    if level_weights is None:
        level_weights = [1.0 / len(levels)] * len(levels)
    return levels, weighted(levels, level_weights)


def normalized_regret(
    selected_utility: float | None, best_utility: float, worst_utility: float
) -> float | None:
    if selected_utility is None:
        return None
    span = best_utility - worst_utility
    if span <= 1e-15:
        return 0.0
    return clip((best_utility - selected_utility) / span)


def control_utility_from_outcomes(
    latency_ms: float, error_rate: float, recovery_s: float, resource_index: float
) -> float:
    """Nonlinear synthetic holdout score; it is not the dissertation Q."""
    components = (
        clip((800.0 - latency_ms) / 550.0),
        clip((0.060 - error_rate) / 0.060),
        clip((300.0 - recovery_s) / 300.0),
        clip(1.0 - resource_index),
    )
    weights = (0.30, 0.30, 0.20, 0.20)
    return math.exp(
        sum(weight * math.log(max(value, 1e-12)) for value, weight in zip(components, weights))
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_control_alternatives(
    criterion_weights: Sequence[float],
) -> tuple[dict[str, list[Alternative]], list[dict[str, object]], list[dict[str, object]]]:
    with (INPUTS / "criterion_passports.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        passports = list(csv.DictReader(stream))
    assert [row["criterion"] for row in passports] == ["f1", "f2", "f3"]
    assert all(row["correspondence_function"] == "identity" for row in passports)
    assert all(row["direction"] == "max" for row in passports)
    passport_weights = [float(row["weight"]) for row in passports]
    assert all(
        abs(left - right) <= 1e-15
        for left, right in zip(passport_weights, criterion_weights)
    )

    certificate_rows: list[dict[str, object]] = []
    certificate_result: dict[tuple[str, str], bool] = {}
    checks = (
        "boundary_signature",
        "neighbor_compatibility",
        "trace_elements",
        "mandatory_constraints",
        "transferred_characteristics",
    )
    with (INPUTS / "control_certificates.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            calculated = all(row[name] == "1" for name in checks)
            asserted = row["R"] == "1"
            assert calculated == asserted
            certificate_result[(row["dominator"], row["dominated"])] = calculated
            certificate_rows.append({**row, "calculated_R": int(calculated)})

    grouped_vertices: dict[tuple[str, str], list[dict[str, str]]] = {}
    with (INPUTS / "control_vertices.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped_vertices.setdefault((row["level"], row["alternative"]), []).append(row)

    vertex_calculations: list[dict[str, object]] = []
    result: dict[str, list[Alternative]] = {"A": [], "B": [], "C": []}
    with (INPUTS / "control_alternatives.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["level"], row["alternative"])
            vertices = grouped_vertices[key]
            assert {vertex["vertex"] for vertex in vertices} == {"core", "boundary"}
            lambdas = [float(vertex["lambda"]) for vertex in vertices]
            assert abs(sum(lambdas) - 1.0) <= 1e-15
            p_by_vertex: list[tuple[float, float, float]] = []
            for vertex in vertices:
                raw_x = tuple(float(vertex[f"x{k}"]) for k in range(1, 4))
                assert all(0.0 <= value <= 1.0 for value in raw_x)
                p = raw_x  # mu(x)=x according to all three criterion passports
                p_by_vertex.append(p)  # type: ignore[arg-type]
                vertex_calculations.append(
                    {
                        "level": row["level"],
                        "alternative": row["alternative"],
                        "vertex": vertex["vertex"],
                        "lambda": vertex["lambda"],
                        "x1": f"{raw_x[0]:.12f}",
                        "x2": f"{raw_x[1]:.12f}",
                        "x3": f"{raw_x[2]:.12f}",
                        "p1": f"{p[0]:.12f}",
                        "p2": f"{p[1]:.12f}",
                        "p3": f"{p[2]:.12f}",
                    }
                )
            f = tuple(
                sum(lambdas[index] * p_by_vertex[index][criterion] for index in range(2))
                for criterion in range(3)
            )
            expected_f = tuple(float(row[f"f{k}"]) for k in range(1, 4))
            assert all(abs(actual - expected) <= 5e-13 for actual, expected in zip(f, expected_f))
            dominated_by = row["dominated_by"] or None
            certified = (
                certificate_result.get((dominated_by, row["alternative"]), False)
                if dominated_by
                else False
            )
            assert certified == (row["safe_certificate"] == "1")
            alt = Alternative(
                level=row["level"],
                name=row["alternative"],
                f=f,  # type: ignore[arg-type]
                q0=weighted(f, criterion_weights),
                signature=row["signature"],
                interface_fit=float(row["interface_fit"]),
                latent=f,  # not used for the deterministic control example
                dominated_by=dominated_by,
                safe_certificate=certified,
            )
            result[alt.level].append(alt)
    assert set(grouped_vertices) == {
        (alt.level, alt.name) for group in result.values() for alt in group
    }
    return result, vertex_calculations, certificate_rows


def run_control(config: dict) -> tuple[dict, list[dict[str, object]]]:
    criterion_weights = config["criterion_weights"]
    control_config = config["control"]
    levels = ("A", "B", "C")
    alternatives, vertex_calculations, certificate_rows = load_control_alternatives(
        criterion_weights
    )
    expectations = load_json(INPUTS / "control_expectations.json")
    fronts = {level: local_front(alternatives[level]) for level in levels}
    transfers = {level: transfer_set(alternatives[level]) for level in levels}
    assert {key: [a.name for a in value] for key, value in fronts.items()} == expectations[
        "local_fronts"
    ]
    assert {key: [a.name for a in value] for key, value in transfers.items()} == expectations[
        "transfer_sets"
    ]
    full_space = math.prod(len(alternatives[level]) for level in levels)
    transfer_space = math.prod(len(transfers[level]) for level in levels)
    assert full_space == expectations["full_space_size"] == 27
    assert transfer_space == expectations["transfer_space_size"] == 18

    alt_by_name = {alt.name: alt for group in alternatives.values() for alt in group}
    state_rows: list[dict[str, str]] = []
    with (INPUTS / "control_states.csv").open(encoding="utf-8", newline="") as stream:
        state_rows = list(csv.DictReader(stream))
    state_by_name = {row["state"]: row for row in state_rows}
    with (INPUTS / "control_compatibility.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        compatibility_rows = list(csv.DictReader(stream))
    assert len(compatibility_rows) == full_space
    assert {tuple(row[level] for level in levels) for row in compatibility_rows} == set(
        itertools.product(
            *[[alt.name for alt in alternatives[level]] for level in levels]
        )
    )
    compatible = {
        tuple(row[level] for level in levels): state_by_name[row["state"]]
        for row in compatibility_rows
        if row["compatible"] == "1"
    }
    assert set(state_by_name) == {
        row["state"] for row in compatibility_rows if row["compatible"] == "1"
    }
    assert len(compatible) == expectations["oracle_feasible_states"] == 5

    evaluations: dict[tuple[str, ...], StateEvaluation] = {}
    iteration_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for state, row in compatible.items():
        selected = [alt_by_name[name] for name in state]
        q0_levels, q0 = aggregate_scores(
            [alt.f for alt in selected], criterion_weights, control_config["level_weights"]
        )
        corrected, iterations, residual, history = trace_adjust(
            [alt.f for alt in selected],
            control_config["rho"],
            float(row["trace_bias"]),
            control_config["epsilon"],
            control_config["max_iterations"],
        )
        corrected_levels, q = aggregate_scores(
            corrected, criterion_weights, control_config["level_weights"]
        )
        utility = control_utility_from_outcomes(
            float(row["p95_latency_ms"]),
            float(row["error_rate"]),
            float(row["recovery_s"]),
            float(row["resource_index"]),
        )
        evaluation = StateEvaluation(
            state=state,
            q0=q0,
            q=q,
            vector=tuple(value for vector in corrected for value in vector),
            iterations=iterations,
            residual=residual,
            control_utility=utility,
            trace_bias=float(row["trace_bias"]),
            p95_latency_ms=float(row["p95_latency_ms"]),
            error_rate=float(row["error_rate"]),
            recovery_s=float(row["recovery_s"]),
            resource_index=float(row["resource_index"]),
        )
        evaluations[state] = evaluation
        assert history and history[-1][0] == iterations
        assert all(
            history[index][1]
            <= control_config["rho"] * history[index - 1][1] + 1e-14
            for index in range(1, len(history))
        )
        iteration_rows.extend(
            {
                "state": row["state"],
                "iteration": iteration,
                "residual": f"{iteration_residual:.15e}",
            }
            for iteration, iteration_residual in history
        )
        for level_index, level in enumerate(levels):
            for criterion_index in range(3):
                position_rows.append(
                    {
                        "state": row["state"],
                        "level": level,
                        "criterion": f"f{criterion_index + 1}",
                        "initial": f"{selected[level_index].f[criterion_index]:.12f}",
                        "converged": f"{corrected[level_index][criterion_index]:.12f}",
                        "iterations": iterations,
                        "residual": f"{residual:.3e}",
                    }
                )
        raw_rows.append(
            {
                "state": row["state"],
                "A": state[0],
                "B": state[1],
                "C": state[2],
                "q0_A": f"{q0_levels[0]:.12f}",
                "q0_B": f"{q0_levels[1]:.12f}",
                "q0_C": f"{q0_levels[2]:.12f}",
                "q0": f"{q0:.12f}",
                "adjusted_A": f"{corrected_levels[0]:.12f}",
                "adjusted_B": f"{corrected_levels[1]:.12f}",
                "adjusted_C": f"{corrected_levels[2]:.12f}",
                "q": f"{q:.12f}",
                "trace_bias": row["trace_bias"],
                "iterations": iterations,
                "residual": f"{residual:.3e}",
                "p95_latency_ms": row["p95_latency_ms"],
                "error_rate": row["error_rate"],
                "recovery_s": row["recovery_s"],
                "resource_index": row["resource_index"],
                "control_utility": f"{utility:.12f}",
            }
        )

    oracle_evaluations = list(evaluations.values())
    oracle_front = pareto_front(oracle_evaluations)
    oracle_front_names = sorted(
        next(
            row["state"]
            for row in state_rows
            if tuple(row[level] for level in levels) == item.state
        )
        for item in oracle_front
    )
    assert oracle_front_names == expectations["global_front"], (
        oracle_front_names,
        expectations["global_front"],
    )
    oracle = max(oracle_front, key=lambda item: (item.q, tuple(reversed(item.state))))

    transfer_names = {level: {alt.name for alt in transfers[level]} for level in levels}
    full_candidates = [
        evaluation
        for state, evaluation in evaluations.items()
        if all(state[index] in transfer_names[level] for index, level in enumerate(levels))
    ]
    assert len(full_candidates) == expectations["full_method_feasible_states"] == 3
    full_front = pareto_front(full_candidates)
    full_method = max(full_front, key=lambda item: (item.q, tuple(reversed(item.state))))
    assert full_method.state == oracle.state
    assert {item.state for item in full_front} == {item.state for item in oracle_front}

    front_names = {level: {alt.name for alt in fronts[level]} for level in levels}
    unsafe_candidates = [
        evaluation
        for state, evaluation in evaluations.items()
        if all(state[index] in front_names[level] for index, level in enumerate(levels))
    ]
    unsafe_front = pareto_front(unsafe_candidates)
    unsafe = max(unsafe_front, key=lambda item: (item.q, tuple(reversed(item.state))))
    unsafe_recall = len({x.state for x in unsafe_front} & {x.state for x in oracle_front}) / len(
        oracle_front
    )

    additive = max(oracle_evaluations, key=lambda item: (item.q0, tuple(reversed(item.state))))
    local_greedy = tuple(
        max(alternatives[level], key=lambda item: (item.q0, item.name)).name for level in levels
    )
    assert local_greedy == tuple(expectations["local_greedy_state"])
    assert local_greedy not in compatible
    state_name = {tuple(row[level] for level in levels): row["state"] for row in state_rows}
    assert state_name[additive.state] == expectations["additive_state"]
    assert abs(additive.q0 - expectations["additive_q0"]) <= 5e-13
    assert state_name[unsafe.state] == expectations["unsafe_pareto_state"]
    assert abs(unsafe_recall - expectations["unsafe_pareto_recall"]) <= 1e-12
    assert state_name[full_method.state] == expectations["full_method_state"]
    assert state_name[oracle.state] == expectations["oracle_state"]
    assert abs(full_method.q - expectations["full_method_q"]) <= 5e-7
    assert all(item.residual <= control_config["epsilon"] for item in evaluations.values())

    write_csv(
        RESULTS / "control_alternatives_calculated.csv",
        (
            "level",
            "alternative",
            "f1",
            "f2",
            "f3",
            "q0",
            "local_front",
            "transfer_set",
            "dominated_by",
            "safe_certificate",
        ),
        (
            {
                "level": level,
                "alternative": alt.name,
                "f1": f"{alt.f[0]:.12f}",
                "f2": f"{alt.f[1]:.12f}",
                "f3": f"{alt.f[2]:.12f}",
                "q0": f"{alt.q0:.12f}",
                "local_front": int(alt in fronts[level]),
                "transfer_set": int(alt in transfers[level]),
                "dominated_by": alt.dominated_by or "",
                "safe_certificate": int(alt.safe_certificate),
            }
            for level in levels
            for alt in alternatives[level]
        ),
    )
    write_csv(
        RESULTS / "control_vertices_calculated.csv",
        tuple(vertex_calculations[0].keys()),
        vertex_calculations,
    )
    write_csv(
        RESULTS / "control_certificates_calculated.csv",
        tuple(certificate_rows[0].keys()),
        certificate_rows,
    )
    write_csv(
        RESULTS / "control_states_raw.csv",
        tuple(raw_rows[0].keys()),
        raw_rows,
    )
    write_csv(
        RESULTS / "control_iteration_trace.csv",
        tuple(iteration_rows[0].keys()),
        iteration_rows,
    )
    write_csv(
        RESULTS / "control_positions.csv",
        tuple(position_rows[0].keys()),
        position_rows,
    )
    control_summary = {
        "full_space_size": full_space,
        "transfer_space_size": transfer_space,
        "candidate_reduction": 1.0 - transfer_space / full_space,
        "oracle_feasible_states": len(oracle_evaluations),
        "full_method_feasible_states": len(full_candidates),
        "full_state_reduction": 1.0 - len(full_candidates) / len(oracle_evaluations),
        "local_fronts": {level: [alt.name for alt in fronts[level]] for level in levels},
        "transfer_sets": {level: [alt.name for alt in transfers[level]] for level in levels},
        "global_front": oracle_front_names,
        "calculation_chain": "x -> p=mu(x) -> lambda -> F -> Q0 -> trace -> Q",
        "local_greedy": {
            "state": list(local_greedy),
            "feasible": False,
        },
        "feasible_additive_no_trace": {
            "state": state_name[additive.state],
            "q0": additive.q0,
            "control_utility": additive.control_utility,
        },
        "unsafe_pareto": {
            "state": state_name[unsafe.state],
            "q": unsafe.q,
            "pareto_recall": unsafe_recall,
        },
        "full_method": {
            "state": state_name[full_method.state],
            "q": full_method.q,
            "control_utility": full_method.control_utility,
            "iterations": full_method.iterations,
            "residual": full_method.residual,
        },
        "oracle_full_model": {
            "state": state_name[oracle.state],
            "q": oracle.q,
            "control_utility": oracle.control_utility,
        },
    }
    return control_summary, raw_rows


def generate_batch_task(
    regime: str,
    replicate: int,
    seed: int,
    config: dict,
    criterion_weights: Sequence[float],
) -> tuple[dict[str, list[Alternative]], dict[str, object]]:
    rng = random.Random(seed)
    levels = [f"L{i + 1}" for i in range(config["levels"])]
    alternatives: dict[str, list[Alternative]] = {}
    for level_index, level in enumerate(levels):
        group: list[Alternative] = []
        for index in range(6):
            focus = (index + level_index) % 3
            f_values = [0.44 + 0.40 * rng.random() for _ in range(3)]
            f_values[focus] = clip(f_values[focus] + 0.12, high=0.97)
            f = tuple(f_values)
            interface_fit = rng.random()
            latent = tuple(
                clip(0.84 * value + 0.16 * (0.35 + 0.60 * rng.random())) for value in f
            )
            group.append(
                Alternative(
                    level=level,
                    name=f"{level}_{index + 1}",
                    f=f,  # type: ignore[arg-type]
                    q0=weighted(f, criterion_weights),
                    signature=str(index % 3),
                    interface_fit=interface_fit,
                    latent=latent,  # type: ignore[arg-type]
                )
            )
        safe_base = group[0]
        safe_f = tuple(clip(value - (0.035 + 0.035 * rng.random())) for value in safe_base.f)
        safe_latent = tuple(clip(value - 0.025) for value in safe_base.latent)
        group.append(
            Alternative(
                level=level,
                name=f"{level}_7",
                f=safe_f,  # type: ignore[arg-type]
                q0=weighted(safe_f, criterion_weights),
                signature=safe_base.signature,
                interface_fit=safe_base.interface_fit,
                latent=safe_latent,  # type: ignore[arg-type]
                dominated_by=safe_base.name,
                safe_certificate=True,
            )
        )
        unsafe_base = group[1]
        unsafe_f = tuple(clip(value - (0.025 + 0.045 * rng.random())) for value in unsafe_base.f)
        unsafe_latent = tuple(clip(value - 0.015) for value in unsafe_base.latent)
        group.append(
            Alternative(
                level=level,
                name=f"{level}_8",
                f=unsafe_f,  # type: ignore[arg-type]
                q0=weighted(unsafe_f, criterion_weights),
                signature=str((int(unsafe_base.signature) + 1) % 3),
                interface_fit=rng.random(),
                latent=unsafe_latent,  # type: ignore[arg-type]
                dominated_by=unsafe_base.name,
                safe_certificate=False,
            )
        )
        assert len(group) == config["alternatives_per_level"] == 8
        assert dominates(safe_base.f, group[6].f)
        assert safe_base.signature == group[6].signature
        assert safe_base.interface_fit == group[6].interface_fit
        assert dominates(unsafe_base.f, group[7].f)
        alternatives[level] = group
    metadata = {
        "regime": regime,
        "replicate": replicate,
        "seed": seed,
        "holdout_seed": (seed * 1103515245 + 12345) % (2**31),
        "holdout_samples": 40,
    }
    return alternatives, metadata


def batch_state_feasible(
    selected: Sequence[Alternative], regime: str, seed: int, conflict_probability: float
) -> bool:
    for edge, (left, right) in enumerate(zip(selected, selected[1:])):
        if left.signature != right.signature:
            return False
        if regime == "conflict":
            key = (
                int(round(left.interface_fit * 1000.0)) * 31
                + int(round(right.interface_fit * 1000.0)) * 17
                + seed * 7
                + edge * 13
            ) % 1000
            if key < int(conflict_probability * 1000):
                return False
    return True


def batch_trace_bias(
    selected: Sequence[Alternative],
    trace_scale: float,
    criterion_weights: Sequence[float],
) -> float:
    if trace_scale == 0.0:
        return 0.0
    pair_values = []
    for left, right in zip(selected, selected[1:]):
        cohesion = 1.0 - abs(left.interface_fit - right.interface_fit)
        left_q = weighted(left.f, criterion_weights)
        right_q = weighted(right.f, criterion_weights)
        pair_values.append(min(left_q, right_q) * (0.45 + 0.55 * cohesion))
    interaction = statistics.fmean(pair_values)
    return trace_scale * (interaction - 0.52)


def batch_control_outcomes(
    selected: Sequence[Alternative], workloads: Sequence[float]
) -> tuple[float, float, float, float, float]:
    """Independent nonlinear simulator evaluated on held-out workload samples."""
    throughput = len(selected) / sum(1.0 / max(alt.latent[0], 1e-9) for alt in selected)
    resilience = min(alt.latent[1] for alt in selected)
    maintainability = statistics.fmean(alt.latent[2] for alt in selected)
    cohesion = statistics.fmean(
        1.0 - abs(left.interface_fit - right.interface_fit)
        for left, right in zip(selected, selected[1:])
    )
    dispersion = statistics.pstdev(alt.latent[0] for alt in selected)
    latencies = [
        1080.0 - 525.0 * throughput - 145.0 * cohesion + 65.0 * workload
        for workload in workloads
    ]
    ordered_latencies = sorted(latencies)
    p95_latency_ms = ordered_latencies[math.ceil(0.95 * len(ordered_latencies)) - 1]
    error_rate = statistics.fmean(
        0.082 * (1.0 - resilience) ** 1.35
        + 0.019 * (1.0 - cohesion)
        + 0.003 * workload
        for workload in workloads
    )
    recovery_s = statistics.fmean(
        385.0 * (1.0 - math.sqrt(resilience * maintainability)) + 42.0 * workload
        for workload in workloads
    )
    resource_index = statistics.fmean(
        clip(
            0.25
            + 0.52 * (1.0 - maintainability)
            + 0.10 * dispersion
            + 0.045 * workload
        )
        for workload in workloads
    )
    utility = control_utility_from_outcomes(
        p95_latency_ms, error_rate, recovery_s, resource_index
    )
    return utility, p95_latency_ms, error_rate, recovery_s, resource_index


def batch_evaluate_state(
    selected: Sequence[Alternative],
    regime_config: dict,
    batch_config: dict,
    criterion_weights: Sequence[float],
    workloads: Sequence[float],
) -> StateEvaluation:
    q0, q, corrected, iterations, residual, bias = batch_model_scores(
        selected, regime_config, batch_config, criterion_weights
    )
    utility, latency, error_rate, recovery, resource = batch_control_outcomes(
        selected, workloads
    )
    return StateEvaluation(
        state=tuple(alt.name for alt in selected),
        q0=q0,
        q=q,
        vector=tuple(value for vector in corrected for value in vector),
        iterations=iterations,
        residual=residual,
        control_utility=utility,
        trace_bias=bias,
        p95_latency_ms=latency,
        error_rate=error_rate,
        recovery_s=recovery,
        resource_index=resource,
    )


def batch_model_scores(
    selected: Sequence[Alternative],
    regime_config: dict,
    batch_config: dict,
    criterion_weights: Sequence[float],
) -> tuple[
    float,
    float,
    tuple[tuple[float, ...], ...],
    int,
    float,
    float,
]:
    _, q0 = aggregate_scores([alt.f for alt in selected], criterion_weights)
    bias = batch_trace_bias(selected, regime_config["trace_scale"], criterion_weights)
    corrected, iterations, residual, _ = trace_adjust(
        [alt.f for alt in selected],
        regime_config["rho"],
        bias,
        batch_config["epsilon"],
        batch_config["max_iterations"],
    )
    _, q = aggregate_scores(corrected, criterion_weights)
    return q0, q, corrected, iterations, residual, bias


def choose_max(evaluations: Sequence[StateEvaluation], field: str) -> StateEvaluation:
    return max(evaluations, key=lambda item: (getattr(item, field), tuple(reversed(item.state))))


def sensitivity_scenarios(
    criterion_weights: Sequence[float], regime_config: dict
) -> list[tuple[str, str, int, list[float], dict]]:
    scenarios: list[tuple[str, str, int, list[float], dict]] = []
    for percent in (-20, -10, 10, 20):
        factor = 1.0 + percent / 100.0
        for criterion_index in range(3):
            changed = list(criterion_weights)
            changed[criterion_index] *= factor
            total = sum(changed)
            normalized = [value / total for value in changed]
            scenarios.append(
                (
                    f"weight_f{criterion_index + 1}_{percent:+d}",
                    "criterion_weight",
                    percent,
                    normalized,
                    dict(regime_config),
                )
            )
        rho_config = dict(regime_config)
        rho_config["rho"] = min(0.99, regime_config["rho"] * factor)
        scenarios.append(
            (
                f"rho_{percent:+d}",
                "rho",
                percent,
                list(criterion_weights),
                rho_config,
            )
        )
        scale_config = dict(regime_config)
        scale_config["trace_scale"] = regime_config["trace_scale"] * factor
        scenarios.append(
            (
                f"trace_scale_{percent:+d}",
                "trace_scale",
                percent,
                list(criterion_weights),
                scale_config,
            )
        )
    return scenarios


def run_task_sensitivity(
    regime: str,
    replicate: int,
    seed: int,
    full_states: Sequence[Sequence[Alternative]],
    baseline_state: tuple[str, ...],
    criterion_weights: Sequence[float],
    regime_config: dict,
    batch_config: dict,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, parameter_group, percent, weights, changed_config in sensitivity_scenarios(
        criterion_weights, regime_config
    ):
        scored: list[tuple[float, tuple[str, ...], int, float]] = []
        for state in full_states:
            _, q, _, iterations, residual, _ = batch_model_scores(
                state, changed_config, batch_config, weights
            )
            scored.append((q, tuple(alt.name for alt in state), iterations, residual))
        choice = max(scored, key=lambda item: (item[0], tuple(reversed(item[1]))))
        assert choice[3] <= batch_config["epsilon"]
        rows.append(
            {
                "regime": regime,
                "replicate": replicate,
                "seed": seed,
                "scenario": scenario,
                "parameter_group": parameter_group,
                "perturbation_percent": percent,
                "baseline_state": "|".join(baseline_state),
                "selected_state": "|".join(choice[1]),
                "choice_stable": int(choice[1] == baseline_state),
                "selected_q": f"{choice[0]:.12f}",
                "iterations": choice[2],
            }
        )
    return rows


def run_batch(
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    batch_config = config["batch"]
    criterion_weights = config["criterion_weights"]
    master = random.Random(config["master_seed"])
    raw_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    task_seeds = [
        master.randrange(1, 2**31)
        for _ in range(batch_config["replicates_per_regime"])
    ]
    fingerprints: dict[int, tuple] = {}
    for regime, regime_config in batch_config["regimes"].items():
        for replicate in range(1, batch_config["replicates_per_regime"] + 1):
            seed = task_seeds[replicate - 1]
            alternatives, metadata = generate_batch_task(
                regime, replicate, seed, batch_config, criterion_weights
            )
            levels = sorted(alternatives)
            fingerprint = tuple(
                (
                    alt.level,
                    alt.name,
                    alt.f,
                    alt.signature,
                    alt.interface_fit,
                    alt.latent,
                )
                for level in levels
                for alt in alternatives[level]
            )
            if replicate in fingerprints:
                assert fingerprint == fingerprints[replicate]
            else:
                fingerprints[replicate] = fingerprint
            holdout_rng = random.Random(int(metadata["holdout_seed"]))
            workloads = tuple(
                0.25 + 0.70 * holdout_rng.random()
                for _ in range(int(metadata["holdout_samples"]))
            )
            for level in levels:
                fronts = local_front(alternatives[level])
                transfers = transfer_set(alternatives[level])
                for alt in alternatives[level]:
                    instance_rows.append(
                        {
                            "regime": regime,
                            "replicate": replicate,
                            "seed": seed,
                            "level": level,
                            "alternative": alt.name,
                            "f1": f"{alt.f[0]:.12f}",
                            "f2": f"{alt.f[1]:.12f}",
                            "f3": f"{alt.f[2]:.12f}",
                            "q0": f"{alt.q0:.12f}",
                            "signature": alt.signature,
                            "interface_fit": f"{alt.interface_fit:.12f}",
                            "latent_performance": f"{alt.latent[0]:.12f}",
                            "latent_reliability": f"{alt.latent[1]:.12f}",
                            "latent_maintainability": f"{alt.latent[2]:.12f}",
                            "dominated_by": alt.dominated_by or "",
                            "safe_certificate": int(alt.safe_certificate),
                            "local_front": int(alt in fronts),
                            "transfer_set": int(alt in transfers),
                        }
                    )
            all_states = list(itertools.product(*(alternatives[level] for level in levels)))
            feasible_states = [
                state
                for state in all_states
                if batch_state_feasible(
                    state, regime, seed, regime_config["conflict_probability"]
                )
            ]
            assert feasible_states, f"no feasible state for {regime}/{replicate}"
            evaluations = {
                tuple(alt.name for alt in state): batch_evaluate_state(
                    state, regime_config, batch_config, criterion_weights, workloads
                )
                for state in feasible_states
            }
            assert all(
                item.residual <= batch_config["epsilon"] for item in evaluations.values()
            )
            oracle_evaluations = list(evaluations.values())
            oracle_front = pareto_front(oracle_evaluations)
            oracle = choose_max(oracle_front, "q")
            holdout_best = max(x.control_utility for x in oracle_evaluations)
            holdout_worst = min(x.control_utility for x in oracle_evaluations)
            oracle_front_states = {x.state for x in oracle_front}
            full_space = len(all_states)

            fronts_by_level = {level: local_front(alternatives[level]) for level in levels}
            transfers_by_level = {level: transfer_set(alternatives[level]) for level in levels}
            candidate_sets: dict[str, list[StateEvaluation]] = {}
            candidate_space_sizes: dict[str, int] = {}

            greedy_state = tuple(
                max(alternatives[level], key=lambda alt: (alt.q0, alt.name)).name
                for level in levels
            )
            candidate_space_sizes["local-greedy"] = 1
            candidate_sets["local-greedy"] = (
                [evaluations[greedy_state]] if greedy_state in evaluations else []
            )
            candidate_space_sizes["feasible-additive-no-trace"] = full_space
            candidate_sets["feasible-additive-no-trace"] = oracle_evaluations

            unsafe_names = {level: {a.name for a in fronts_by_level[level]} for level in levels}
            unsafe_product = math.prod(len(fronts_by_level[level]) for level in levels)
            candidate_space_sizes["unsafe-pareto"] = unsafe_product
            candidate_sets["unsafe-pareto"] = [
                evaluation
                for state, evaluation in evaluations.items()
                if all(state[i] in unsafe_names[level] for i, level in enumerate(levels))
            ]

            transfer_names = {
                level: {a.name for a in transfers_by_level[level]} for level in levels
            }
            transfer_product = math.prod(len(transfers_by_level[level]) for level in levels)
            candidate_space_sizes["full-method"] = transfer_product
            candidate_sets["full-method"] = [
                evaluation
                for state, evaluation in evaluations.items()
                if all(state[i] in transfer_names[level] for i, level in enumerate(levels))
            ]
            candidate_space_sizes["oracle-full-model"] = full_space
            candidate_sets["oracle-full-model"] = oracle_evaluations

            selected: dict[str, StateEvaluation | None] = {}
            method_fronts: dict[str, list[StateEvaluation]] = {}
            method_fronts["local-greedy"] = candidate_sets["local-greedy"]
            selected["local-greedy"] = (
                candidate_sets["local-greedy"][0] if candidate_sets["local-greedy"] else None
            )
            method_fronts["feasible-additive-no-trace"] = [
                choose_max(candidate_sets["feasible-additive-no-trace"], "q0")
            ]
            selected["feasible-additive-no-trace"] = method_fronts[
                "feasible-additive-no-trace"
            ][0]
            for method in ("unsafe-pareto", "full-method", "oracle-full-model"):
                method_fronts[method] = (
                    pareto_front(candidate_sets[method]) if candidate_sets[method] else []
                )
                selected[method] = (
                    choose_max(method_fronts[method], "q") if method_fronts[method] else None
                )

            assert selected["full-method"] is not None
            assert selected["oracle-full-model"] is not None
            assert selected["full-method"].state == selected["oracle-full-model"].state
            assert abs(selected["full-method"].q - selected["oracle-full-model"].q) <= 1e-12
            assert {item.state for item in method_fronts["full-method"]} == {
                item.state for item in method_fronts["oracle-full-model"]
            }
            if regime == "no":
                assert selected["feasible-additive-no-trace"] is not None
                assert selected["feasible-additive-no-trace"].state == selected["full-method"].state
                assert abs(selected["full-method"].q - selected["full-method"].q0) <= 1e-14

            task_rows.append(
                {
                    "regime": regime,
                    "replicate": replicate,
                    "seed": seed,
                    "holdout_seed": metadata["holdout_seed"],
                    "holdout_samples": metadata["holdout_samples"],
                    "holdout_workload_mean": f"{statistics.fmean(workloads):.12f}",
                    "full_space": full_space,
                    "oracle_feasible_states": len(oracle_evaluations),
                    "oracle_pareto_states": len(oracle_front),
                    "transfer_space": transfer_product,
                    "full_method_feasible_states": len(candidate_sets["full-method"]),
                }
            )
            full_state_objects = [
                state
                for state in feasible_states
                if all(
                    state[index].name in transfer_names[level]
                    for index, level in enumerate(levels)
                )
            ]
            sensitivity_rows.extend(
                run_task_sensitivity(
                    regime,
                    replicate,
                    seed,
                    full_state_objects,
                    selected["full-method"].state,
                    criterion_weights,
                    regime_config,
                    batch_config,
                )
            )
            for method in METHODS:
                choice = selected[method]
                feasible = choice is not None
                method_front_states = {x.state for x in method_fronts[method]}
                pareto_recall = (
                    len(method_front_states & oracle_front_states) / len(oracle_front_states)
                    if method in {"unsafe-pareto", "full-method", "oracle-full-model"}
                    else None
                )
                evaluated_count = len(candidate_sets[method])
                uses_trace = method in {
                    "unsafe-pareto",
                    "full-method",
                    "oracle-full-model",
                }
                trace_evaluations = evaluated_count if uses_trace else 0
                mean_iterations = (
                    statistics.fmean(x.iterations for x in candidate_sets[method])
                    if uses_trace and candidate_sets[method]
                    else 0.0
                )
                regret = normalized_regret(
                    choice.control_utility if choice else None,
                    holdout_best,
                    holdout_worst,
                )
                raw_rows.append(
                    {
                        "regime": regime,
                        "replicate": replicate,
                        "seed": seed,
                        "method": method,
                        "selected_state": "|".join(choice.state) if choice else "|".join(greedy_state),
                        "feasible": int(feasible),
                        "top1_agreement": int(feasible and choice.state == oracle.state),
                        "selected_q0": f"{choice.q0:.12f}" if choice else "",
                        "selected_q": (
                            f"{choice.q:.12f}" if choice and uses_trace else ""
                        ),
                        "selected_control_utility": (
                            f"{choice.control_utility:.12f}" if choice else ""
                        ),
                        "selected_p95_latency_ms": (
                            f"{choice.p95_latency_ms:.12f}" if choice else ""
                        ),
                        "selected_error_rate": f"{choice.error_rate:.12f}" if choice else "",
                        "selected_recovery_s": f"{choice.recovery_s:.12f}" if choice else "",
                        "selected_resource_index": (
                            f"{choice.resource_index:.12f}" if choice else ""
                        ),
                        "normalized_regret": f"{regret:.12f}" if regret is not None else "",
                        "pareto_recall": (
                            f"{pareto_recall:.12f}" if pareto_recall is not None else ""
                        ),
                        "candidate_space": candidate_space_sizes[method],
                        "candidate_reduction": f"{1.0 - candidate_space_sizes[method] / full_space:.12f}",
                        "compatible_states_considered": evaluated_count,
                        "compatible_state_reduction": f"{1.0 - evaluated_count / len(oracle_evaluations):.12f}",
                        "trace_evaluations": trace_evaluations,
                        "trace_evaluation_reduction": (
                            f"{1.0 - trace_evaluations / len(oracle_evaluations):.12f}"
                            if uses_trace
                            else ""
                        ),
                        "selected_iterations": choice.iterations if choice and uses_trace else 0,
                        "mean_convergence_iterations": f"{mean_iterations:.6f}",
                        "oracle_state": "|".join(oracle.state),
                        "oracle_q": f"{oracle.q:.12f}",
                        "holdout_best_utility": f"{holdout_best:.12f}",
                    }
                )

    assert len(raw_rows) == 4 * 30 * len(METHODS) == 600
    write_csv(RESULTS / "batch_instances.csv", tuple(instance_rows[0].keys()), instance_rows)
    write_csv(RESULTS / "batch_tasks.csv", tuple(task_rows[0].keys()), task_rows)
    write_csv(RESULTS / "batch_raw.csv", tuple(raw_rows[0].keys()), raw_rows)
    write_csv(
        RESULTS / "batch_sensitivity_raw.csv",
        tuple(sensitivity_rows[0].keys()),
        sensitivity_rows,
    )
    return raw_rows, task_rows, sensitivity_rows


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: Sequence[float], seed: int, resamples: int = 5000
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize_batch(
    raw_rows: Sequence[dict[str, object]], master_seed: int
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for regime_index, regime in enumerate(("no", "medium", "strong", "conflict")):
        for method_index, method in enumerate(METHODS):
            group = [row for row in raw_rows if row["regime"] == regime and row["method"] == method]
            assert len(group) == 30
            regret = [
                float(row["normalized_regret"])
                for row in group
                if row["normalized_regret"] != ""
            ]
            feasible_n = sum(int(row["feasible"]) for row in group)
            assert len(regret) == feasible_n
            if regret:
                q1 = percentile(regret, 0.25)
                q3 = percentile(regret, 0.75)
                ci_low, ci_high = bootstrap_mean_ci(
                    regret,
                    master_seed + regime_index * 100 + method_index,
                )
                regret_values = {
                    "mean_normalized_regret": f"{statistics.fmean(regret):.6f}",
                    "median_normalized_regret": f"{statistics.median(regret):.6f}",
                    "q1_normalized_regret": f"{q1:.6f}",
                    "q3_normalized_regret": f"{q3:.6f}",
                    "iqr_normalized_regret": f"{q3 - q1:.6f}",
                    "sd_normalized_regret": (
                        f"{statistics.stdev(regret):.6f}" if len(regret) > 1 else "0.000000"
                    ),
                    "bootstrap_ci95_low": f"{ci_low:.6f}",
                    "bootstrap_ci95_high": f"{ci_high:.6f}",
                }
            else:
                regret_values = {
                    "mean_normalized_regret": "",
                    "median_normalized_regret": "",
                    "q1_normalized_regret": "",
                    "q3_normalized_regret": "",
                    "iqr_normalized_regret": "",
                    "sd_normalized_regret": "",
                    "bootstrap_ci95_low": "",
                    "bootstrap_ci95_high": "",
                }
            recall_values = [
                float(row["pareto_recall"])
                for row in group
                if row["pareto_recall"] != ""
            ]
            outcome_columns = (
                "selected_p95_latency_ms",
                "selected_error_rate",
                "selected_recovery_s",
                "selected_resource_index",
            )
            outcome_means: dict[str, str] = {}
            for column in outcome_columns:
                values = [float(row[column]) for row in group if row[column] != ""]
                assert len(values) == feasible_n
                outcome_means[f"mean_{column.removeprefix('selected_')}"] = (
                    f"{statistics.fmean(values):.6f}" if values else ""
                )
            summaries.append(
                {
                    "regime": regime,
                    "method": method,
                    "n": len(group),
                    "feasible_n": feasible_n,
                    "feasibility_rate": f"{statistics.fmean(int(row['feasible']) for row in group):.6f}",
                    "top1_agreement_rate": f"{statistics.fmean(int(row['top1_agreement']) for row in group):.6f}",
                    **regret_values,
                    "mean_pareto_recall": (
                        f"{statistics.fmean(recall_values):.6f}" if recall_values else ""
                    ),
                    "mean_candidate_reduction": f"{statistics.fmean(float(row['candidate_reduction']) for row in group):.6f}",
                    "mean_compatible_state_reduction": f"{statistics.fmean(float(row['compatible_state_reduction']) for row in group):.6f}",
                    "mean_trace_evaluation_reduction": (
                        f"{statistics.fmean(float(row['trace_evaluation_reduction']) for row in group if row['trace_evaluation_reduction'] != ''):.6f}"
                        if any(row["trace_evaluation_reduction"] != "" for row in group)
                        else ""
                    ),
                    "mean_convergence_iterations": f"{statistics.fmean(float(row['mean_convergence_iterations']) for row in group):.6f}",
                    **outcome_means,
                }
            )
    write_csv(RESULTS / "batch_summary.csv", tuple(summaries[0].keys()), summaries)
    return summaries


def compare_with_full_method(
    raw_rows: Sequence[dict[str, object]], master_seed: int
) -> list[dict[str, object]]:
    """Paired descriptive comparison on exactly the same 30 instances."""
    comparisons: list[dict[str, object]] = []
    baselines = ("local-greedy", "feasible-additive-no-trace", "unsafe-pareto")
    for regime_index, regime in enumerate(("no", "medium", "strong", "conflict")):
        by_key = {
            (int(row["replicate"]), str(row["method"])): row
            for row in raw_rows
            if row["regime"] == regime
        }
        for baseline_index, baseline in enumerate(baselines):
            differences: list[float] = []
            top1_lifts: list[int] = []
            feasibility_lifts: list[int] = []
            wins = ties = losses = 0
            for replicate in range(1, 31):
                full = by_key[(replicate, "full-method")]
                simple = by_key[(replicate, baseline)]
                top1_lifts.append(
                    int(full["top1_agreement"]) - int(simple["top1_agreement"])
                )
                feasibility_lifts.append(int(full["feasible"]) - int(simple["feasible"]))
                if simple["normalized_regret"] != "" and full["normalized_regret"] != "":
                    difference = float(simple["normalized_regret"]) - float(
                        full["normalized_regret"]
                    )
                    differences.append(difference)
                    if difference > 1e-12:
                        wins += 1
                    elif difference < -1e-12:
                        losses += 1
                    else:
                        ties += 1
            if differences:
                mean_difference = statistics.fmean(differences)
                ci_low, ci_high = bootstrap_mean_ci(
                    differences,
                    master_seed + 1000 + regime_index * 100 + baseline_index,
                )
            else:
                mean_difference = ci_low = ci_high = None
            comparisons.append(
                {
                    "regime": regime,
                    "baseline": baseline,
                    "n_paired": len(differences),
                    "excluded_infeasible": 30 - len(differences),
                    "mean_regret_reduction": (
                        f"{mean_difference:.6f}" if mean_difference is not None else ""
                    ),
                    "bootstrap_ci95_low": f"{ci_low:.6f}" if ci_low is not None else "",
                    "bootstrap_ci95_high": f"{ci_high:.6f}" if ci_high is not None else "",
                    "full_better_count": wins,
                    "tie_count": ties,
                    "full_worse_count": losses,
                    "top1_rate_lift": f"{statistics.fmean(top1_lifts):.6f}",
                    "feasibility_rate_lift": f"{statistics.fmean(feasibility_lifts):.6f}",
                }
            )
    write_csv(
        RESULTS / "batch_pairwise.csv",
        tuple(comparisons[0].keys()),
        comparisons,
    )
    return comparisons


def summarize_sensitivity(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    groups = ("criterion_weight", "rho", "trace_scale")
    for regime in ("no", "medium", "strong", "conflict"):
        regime_rows = [row for row in rows if row["regime"] == regime]
        assert len(regime_rows) == 30 * 20
        for parameter_group in groups:
            for percent in (-20, -10, 10, 20):
                selected = [
                    row
                    for row in regime_rows
                    if row["parameter_group"] == parameter_group
                    and int(row["perturbation_percent"]) == percent
                ]
                expected = 90 if parameter_group == "criterion_weight" else 30
                assert len(selected) == expected
                result.append(
                    {
                        "regime": regime,
                        "parameter_group": parameter_group,
                        "perturbation_percent": percent,
                        "n": len(selected),
                        "choice_stability": f"{statistics.fmean(int(row['choice_stable']) for row in selected):.6f}",
                    }
                )
        result.append(
            {
                "regime": regime,
                "parameter_group": "all",
                "perturbation_percent": "all",
                "n": len(regime_rows),
                "choice_stability": f"{statistics.fmean(int(row['choice_stable']) for row in regime_rows):.6f}",
            }
        )
    write_csv(
        RESULTS / "batch_sensitivity_summary.csv",
        tuple(result[0].keys()),
        result,
    )
    return result


def write_manuscript_tables(
    control: dict,
    summaries: Sequence[dict[str, object]],
    sensitivity: Sequence[dict[str, object]],
) -> None:
    lines = [
        "# Таблицы воспроизводимого эксперимента",
        "",
        "Файл сформирован `run_experiment.py`; ручное редактирование результатов не требуется.",
        "",
        "## Контрольный пример",
        "",
        "| Показатель | Значение |",
        "|---|---:|",
        f"| Исходных комбинаций | {control['full_space_size']} |",
        f"| Комбинаций после сертифицированного сокращения | {control['transfer_space_size']} |",
        f"| Допустимых состояний при полном переборе | {control['oracle_feasible_states']} |",
        f"| Допустимых состояний для полного метода | {control['full_method_feasible_states']} |",
        f"| Локально-жадная комбинация | {' + '.join(control['local_greedy']['state'])}, недопустима |",
        f"| Аддитивный выбор без трассировки | {control['feasible_additive_no_trace']['state']}, Q0={control['feasible_additive_no_trace']['q0']:.6f} |",
        f"| Небезопасный Парето-отсев | {control['unsafe_pareto']['state']}, полнота={control['unsafe_pareto']['pareto_recall']:.6f} |",
        f"| Предложенный процесс | {control['full_method']['state']}, Q={control['full_method']['q']:.6f} |",
        f"| Эталонный полный перебор | {control['oracle_full_model']['state']}, Q={control['oracle_full_model']['q']:.6f} |",
        "",
        "## Пакетная серия: ключевые показатели",
        "",
        "| Режим | Процедура | Допустимость | Совпадение top-1 | Средний regret | Полнота фронта | Расчётное сокращение оцениваемых состояний |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['regime']} | {row['method']} | {row['feasibility_rate']} | "
            f"{row['top1_agreement_rate']} | {row['mean_normalized_regret']} | "
            f"{row['mean_pareto_recall'] or '—'} | {row['mean_compatible_state_reduction']} |"
        )
    lines.extend(
        [
            "",
            "Средний `normalized regret` рассчитан только для допустимых рекомендаций по "
            "внецелевой синтетической holdout-полезности, не входящей непосредственно в Q. "
            "Её латентные характеристики связаны с тем же генератором, поэтому это проверка "
            "согласованности, а не независимая внешняя валидация. Для недопустимой "
            "рекомендации значение не определяется. "
            "Полнота фронта не приводится для процедур, возвращающих единственный вариант.",
            "Приведённое сокращение — расчётная разность размеров множеств. Текущий скрипт "
            "сначала оценивает референтное множество и затем фильтрует готовые результаты; "
            "поэтому показатель не является измеренным ускорением запуска.",
            "",
            "## Внецелевые синтетические исходы выбранных состояний",
            "",
            "| Режим | Процедура | p95 задержки, мс | Доля ошибок | Восстановление, с | Ресурсный индекс |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        if row["method"] in {"feasible-additive-no-trace", "full-method"}:
            lines.append(
                f"| {row['regime']} | {row['method']} | {row['mean_p95_latency_ms']} | "
                f"{row['mean_error_rate']} | {row['mean_recovery_s']} | "
                f"{row['mean_resource_index']} |"
            )
    lines.extend(
        [
            "",
            "Показатели рассчитаны на 40 holdout-значениях нагрузки для каждого "
            "синтетического экземпляра и не входят непосредственно в формулу Q.",
            "",
            "## Анализ чувствительности полного метода",
            "",
            "| Режим | Число возмущённых расчётов | Устойчивость выбора |",
            "|---|---:|---:|",
        ]
    )
    for row in sensitivity:
        if row["parameter_group"] == "all":
            lines.append(
                f"| {row['regime']} | {row['n']} | {row['choice_stability']} |"
            )
    lines.extend(
        [
            "",
            "Проверены отдельные изменения каждого веса критерия, коэффициента rho и "
            "масштаба межуровневого влияния на -20, -10, +10 и +20 процентов. Детализация "
            "приведена в `batch_sensitivity_summary.csv`.",
            "",
        ]
    )
    (RESULTS / "manuscript_tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(ROOT / "config.json")
    RESULTS.mkdir(parents=True, exist_ok=True)
    control_summary, _ = run_control(config)
    batch_raw, _, sensitivity_raw = run_batch(config)
    batch_summary = summarize_batch(batch_raw, config["master_seed"])
    batch_pairwise = compare_with_full_method(batch_raw, config["master_seed"])
    sensitivity_summary = summarize_sensitivity(sensitivity_raw)
    write_manuscript_tables(control_summary, batch_summary, sensitivity_summary)
    summary = {
        "schema_version": config["schema_version"],
        "master_seed": config["master_seed"],
        "control": control_summary,
        "batch": {
            "regimes": list(config["batch"]["regimes"]),
            "replicates_per_regime": config["batch"]["replicates_per_regime"],
            "tasks": 4 * config["batch"]["replicates_per_regime"],
            "method_runs": len(batch_raw),
            "summary": batch_summary,
            "paired_comparison": batch_pairwise,
            "sensitivity": sensitivity_summary,
        },
        "assertions": {
            "control_x_p_lambda_F_Q_chain_matches_expected_values": "passed",
            "control_five_part_replacement_certificates_match_R": "passed",
            "control_local_greedy_is_infeasible": "passed",
            "control_unsafe_pareto_recall_is_one_third": "passed",
            "control_trace_changes_S3_to_S2": "passed",
            "safe_full_global_front_equals_oracle_all_120_tasks": "passed",
            "safe_full_top1_equals_oracle_all_120_tasks": "passed",
            "same_30_generated_instances_used_in_all_four_regimes": "passed",
            "negative_control_no_trace_equals_additive": "passed",
            "all_trace_iterations_converged": "passed"
        },
    }
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        "OK: control example; 120 batch tasks; 600 method runs; "
        "all reproducibility assertions passed"
    )


if __name__ == "__main__":
    main()
