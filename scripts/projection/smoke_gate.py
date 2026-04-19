#!/usr/bin/env python3
"""Rigid M2 → M3 gate.

Two classes of checks, post-review policy:

  HARD_GATES   — block M3 (exit 1 if any fails)
    * retrieval top-5 ratio ≥ cfg.gate_min_retrieval_top5_fraction_of_baseline
    * reconstruction MSE ratio ≤ cfg.gate_max_reconstruction_mse_ratio

  DIAGNOSTICS  — warn only (never fail the exit code)
    * identity probe accuracy (warning_threshold / error_threshold)
    * on-manifold drift (warning_threshold)

A run that passes both hard gates is green for M3 even if a diagnostic
is flagged; the diagnostic is reported alongside the gate summary so a
reader knows the latent is a little less discriminative than ideal
(or the decoder a little off-manifold), and the paper records that
transparently.
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import logging
import os
import sys
from typing import Any, Dict

import yaml

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.projection import ProjectionConfig


def _load_cfg(path: str) -> ProjectionConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    fields = {f.name for f in _dc.fields(ProjectionConfig)}
    return ProjectionConfig(**{k: v for k, v in raw.items() if k in fields})


def _get_path(d: Dict[str, Any], keys) -> Any:
    cur = d
    for k in keys:
        if isinstance(cur, list):
            cur = cur[int(k)]
        elif isinstance(cur, dict):
            if k in cur:
                cur = cur[k]
            elif str(k) in cur:
                cur = cur[str(k)]
            else:
                raise KeyError(f"Missing key {k!r} under {keys}")
        else:
            raise TypeError(f"Cannot traverse {type(cur).__name__} at {keys}")
    return cur


def _run_hard_gates(eval_data: dict, cfg: ProjectionConfig) -> Dict[str, dict]:
    HARD_GATES = {
        "retrieval_top5_ratio": {
            "path": ["retrieval", "projected_fraction_of_baseline_topk", "5"],
            "min": cfg.gate_min_retrieval_top5_fraction_of_baseline,
            "description": "Projected retrieval top-5 overlap with baseline top-5",
        },
        "reconstruction_mse_ratio": {
            "path": ["reconstruction", "mse_ratio"],
            "max": cfg.gate_max_reconstruction_mse_ratio,
            "description": "Reconstruction MSE / var(h) on held-out queries",
        },
    }
    out: Dict[str, dict] = {}
    for name, gate in HARD_GATES.items():
        try:
            value = float(_get_path(eval_data, gate["path"]))
        except Exception as exc:
            out[name] = {
                "passed": False, "value": None, "error": str(exc),
                "description": gate["description"],
            }
            continue
        if "min" in gate:
            passed = value >= gate["min"]
            threshold = gate["min"]
            kind = "min"
        else:
            passed = value <= gate["max"]
            threshold = gate["max"]
            kind = "max"
        out[name] = {
            "passed": bool(passed),
            "value": value,
            "threshold": threshold,
            "kind": kind,
            "description": gate["description"],
        }
    return out


def _run_diagnostics(eval_data: dict, cfg: ProjectionConfig) -> Dict[str, dict]:
    DIAGNOSTICS = {
        "identity_probe_accuracy": {
            "path": ["identity_probe", "accuracy"],
            "warning_below": cfg.identity_probe_warning_threshold,
            "error_below": cfg.identity_probe_error_threshold,
            "description": "Identity probe accuracy (diagnostic, not gated)",
        },
        "on_manifold_drift": {
            "path": ["on_manifold_drift"],
            "warning_above": cfg.on_manifold_drift_warning_threshold,
            "description": "Ψ(Φ(h)) drift from h distribution (diagnostic)",
        },
    }
    out: Dict[str, dict] = {}
    for name, diag in DIAGNOSTICS.items():
        try:
            value = float(_get_path(eval_data, diag["path"]))
        except Exception as exc:
            out[name] = {
                "status": "ERROR", "value": None, "error": str(exc),
                "description": diag["description"],
            }
            continue
        if "warning_below" in diag:
            if value < diag.get("error_below", float("-inf")):
                status = "ERROR"
            elif value < diag["warning_below"]:
                status = "WARN"
            else:
                status = "OK"
            threshold_desc = (
                f"warning below {diag['warning_below']}"
                + (f", error below {diag['error_below']}" if "error_below" in diag else "")
            )
        else:
            status = "WARN" if value > diag["warning_above"] else "OK"
            threshold_desc = f"warning above {diag['warning_above']}"
        out[name] = {
            "status": status,
            "value": value,
            "threshold_description": threshold_desc,
            "description": diag["description"],
        }
    return out


def _format_report(hard: Dict[str, dict], diag: Dict[str, dict],
                   all_hard_passed: bool) -> str:
    lines = []
    lines.append("\n=== HARD GATES ===")
    for name, r in hard.items():
        mark = "PASS" if r.get("passed") else "FAIL"
        if r.get("value") is None:
            lines.append(f"[{mark}] {name}: ERROR ({r.get('error')})")
            continue
        kind = r.get("kind", "min")
        lines.append(
            f"[{mark}] {name}: {r['value']:.4f}  ({kind} {r['threshold']})"
        )
    lines.append("\n=== DIAGNOSTICS (informational) ===")
    for name, r in diag.items():
        status = r.get("status", "?")
        if r.get("value") is None:
            lines.append(f"[{status}] {name}: ERROR ({r.get('error')})")
            continue
        lines.append(
            f"[{status:<4}] {name}: {r['value']:.4f}  ({r.get('threshold_description')})"
        )

    lines.append("\n=== RESULT ===")
    if all_hard_passed:
        lines.append("ALL HARD GATES PASSED. M3 can proceed.")
        flagged = [k for k, r in diag.items() if r.get("status") in ("WARN", "ERROR")]
        if flagged:
            lines.append(
                "NOTE: diagnostics flagged but NOT gated: "
                + ", ".join(flagged)
                + ". Latent may be less discriminative than ideal; retrieval "
                "passes, so acceptable for M3."
            )
    else:
        failed = [k for k, r in hard.items() if not r.get("passed")]
        lines.append(
            "HARD GATE FAIL: " + ", ".join(failed)
            + ".\nFIX the failing metric(s) before starting M3."
        )
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--eval", required=True, help="Path to eval.json")
    p.add_argument("--json", action="store_true",
                   help="Print the full report as JSON on stdout.")
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    with open(args.eval) as f:
        eval_data = json.load(f)

    hard = _run_hard_gates(eval_data, cfg)
    diag = _run_diagnostics(eval_data, cfg)
    all_hard_passed = all(r.get("passed") for r in hard.values())

    report = {
        "hard_gates": hard,
        "diagnostics": diag,
        "all_hard_passed": bool(all_hard_passed),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_report(hard, diag, all_hard_passed))
    return 0 if all_hard_passed else 1


if __name__ == "__main__":
    sys.exit(main())
