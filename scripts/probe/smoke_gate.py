#!/usr/bin/env python3
"""Rigid smoke gate: run validators across datasets and fail loudly.

This is the **go/no-go** checkpoint before starting M2. The script
walks ``--output-dir``, runs every applicable validator from
``src/llm_probe/validation.py`` against the thresholds in the supplied
probe config, prints a structured report, and exits with

  - ``0`` iff every gate on every dataset passed
  - ``1`` otherwise (with a per-check breakdown in the report)

There is no "warning" state by design: either the dataset is clean or
the pipeline is not allowed to proceed.

Usage::

    python -m scripts.probe.smoke_gate \\
        --output-dir ./data/llm_probe \\
        --config configs/probe_smoke.yaml
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import logging
import os
import sys

import yaml

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.llm_probe import ProbeConfig, run_smoke_gate


def _load_cfg(path: str) -> ProbeConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    fields = {f.name for f in _dc.fields(ProbeConfig)}
    clean = {k: v for k, v in raw.items() if k in fields}
    return ProbeConfig(**clean)


def _format_report(report: dict) -> str:
    lines = []
    all_ok = report["all_passed"]
    lines.append(f"\n=== SMOKE GATE — {'PASS' if all_ok else 'FAIL'} ===")
    for ds, checks in report["datasets"].items():
        ds_ok = checks.get("_all_passed", False)
        lines.append(f"\n[{ds}] {'PASS' if ds_ok else 'FAIL'}")
        for name, res in checks.items():
            if name == "_all_passed":
                continue
            ok = res.get("passed", False)
            val = res.get("value", "?")
            thr = res.get("threshold", "?")
            mark = "✓" if ok else "✗"
            lines.append(f"  {mark} {name:<28} value={val}  threshold={thr}")
            if not ok:
                details = res.get("details", {})
                # Surface sub-checks when present.
                subs = details.get("subchecks")
                if isinstance(subs, dict):
                    for sname, sres in subs.items():
                        smark = "✓" if sres.get("passed") else "✗"
                        lines.append(
                            f"      {smark} {sname:<24} value={sres.get('value')}  "
                            f"threshold={sres.get('threshold')}"
                        )
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser(description="Rigid smoke gate for probe datasets.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Root directory containing dataset subfolders.")
    p.add_argument("--config", type=str, required=True,
                   help="Path to the probe config (thresholds live here).")
    p.add_argument("--json", action="store_true",
                   help="Print the full report as JSON to stdout.")
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    report = run_smoke_gate(args.output_dir, cfg)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_format_report(report))

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
