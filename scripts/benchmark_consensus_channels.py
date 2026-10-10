"""Evaluate frozen app consensus with explicit membrane channels on all held-out scans."""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from benchmark_all_heldout import evaluate_run, frozen_models, paired_stats, save, sha
from PySide6 import QtCore

from fish_viewer.analysis.controller import (
    AnalysisController,
    default_models,
    default_proposal_model,
    default_refinement_model,
    sensitive_stardist_model,
)


def channel_for(case):
    audit = case["audit"]
    candidates = audit["channel_selection"]["stain_candidates"]
    if candidates:
        return audit[
            "channel"
        ], "Declared Lam/LBR wavelength; ring evidence ranks membrane candidates"
    hints = [r["channel"] for r in audit["channel_scores"] if r.get("metadata_hint")]
    if len(hints) != 1:
        raise ValueError("Ambiguous undeclared cell channel: " + case["image"])
    return hints[
        0
    ], "Single metadata membrane hint; visually reviewed channel projections"


def summarize(rows):
    result = {}
    for split in ("test", "validation", "all_heldout"):
        subset = [r for r in rows if split == "all_heldout" or r["split"] == split]
        volumes = [
            dict(p, group=r["group"])
            for r in subset
            for p in r["volume_pairs"]
            if not p["cut"]
        ]
        distances = [
            dict(p, group=r["group"])
            for r in subset
            for p in r["distance_pairs"]
            if not r["cut_references"][p["reference_id"]]
        ]
        totals = {
            key: sum(r[key] for r in subset)
            for key in (
                "references",
                "complete_references",
                "cells_before_filter",
                "final_cells",
                "signals",
                "matched_before_filter",
                "matched_final",
            )
        }
        totals.update(
            scans=len(subset),
            matched_complete_before=sum(
                sum(
                    not r["cut_references"][m["reference_id"]]
                    for m in r["matches_before_filter"]
                )
                for r in subset
            ),
            matched_complete_final=sum(
                sum(
                    not r["cut_references"][m["reference_id"]]
                    for m in r["matches_final"]
                )
                for r in subset
            ),
            volume=paired_stats(volumes, "manual_um3", "predicted_um3", True),
            distance=paired_stats(distances, "manual_um", "predicted_um"),
        )
        if volumes:
            signed = np.array(
                [(p["predicted_um3"] / p["manual_um3"] - 1) * 100 for p in volumes]
            )
            totals["volume"].update(
                mean_signed_percent=float(signed.mean()),
                median_signed_percent=float(np.median(signed)),
            )
        positive = [p for p in distances if p["manual_um"] > 1e-12]
        totals["distance_relative"] = paired_stats(
            positive, "manual_um", "predicted_um", True
        )
        if distances:
            totals["distance"]["fraction_within_0_2_um"] = float(
                np.mean(
                    [abs(p["predicted_um"] - p["manual_um"]) <= 0.2 for p in distances]
                )
            )
        result[split] = totals
    return result


def publish(run, destination):
    selection = json.loads((run / "selection.json").read_text())
    identity = json.loads((run / "identity.json").read_text())
    rows = []
    for i, case in enumerate(selection["cases"]):
        folder = run / (f"{i + 1:03d}-" + Path(case["image"]).stem)
        row = json.loads((folder / "metrics.json").read_text())
        state = json.loads((folder / "pipeline-result.json").read_text())
        if state["status"] != "complete" or row["image"] != case["image"]:
            raise ValueError("Incomplete or mismatched scan result")
        if row["cell_channel"] != identity["channels"][case["image"]][0]:
            raise ValueError("Evaluation cell channel differs from frozen assignment")
        measured = json.loads(
            (Path(state["report_path"]).parent / state["signal_report"]).read_text()
        )
        with np.load(measured["unfiltered_cell_predictions"], allow_pickle=False) as f:
            metadata = json.loads(str(f["metadata"]))
        if (
            metadata["channel"] != row["cell_channel"]
            or measured["membrane_channel"] != row["cell_channel"]
        ):
            raise ValueError(
                "Cell/signal workers did not use the requested membrane channel"
            )
        consensus = metadata["consensus"]
        if not consensus:
            raise ValueError("Expected both-detector consensus predictions")
        rows.append(row)
    summary = summarize(rows)
    corrected = sum(
        identity["channels"][c["image"]][0] != c["audit"]["channel"]
        for c in selection["cases"]
    )
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "complete",
        "expected_scans": len(selection["cases"]),
        "corrected_cell_channels": corrected,
        "run": str(run.resolve()),
        "identity": identity,
        "summary": summary,
        "results": rows,
        "policy": "All held-out scans, fixed app models and roundness gate. Explicit stain-aware cell channels. Unchanged-channel cell inference reused only with matching checkpoint/manifest hashes and input fingerprints; signals/filtering/distances rerun. Complete-cell one-to-one IoU >=0.5; same-cell/channel signal centers within 0.6 um. Original manual A-B distances only. All outliers retained. Validation influenced model selection and the historical test fold was used in earlier exploration; these are not blind prospective estimates. Sparse labels do not establish precision of unmatched cells.",
    }
    save(destination / "summary.json", report)
    tables = {
        "per-scan": [
            {
                k: r[k]
                for k in (
                    "image",
                    "scan",
                    "split",
                    "group",
                    "cell_channel",
                    "references",
                    "complete_references",
                    "cells_before_filter",
                    "final_cells",
                    "signals",
                    "matched_before_filter",
                    "matched_final",
                    "channel_policy",
                )
            }
            for r in rows
        ],
        "volumes": [
            dict(p, image=r["image"], split=r["split"], group=r["group"])
            for r in rows
            for p in r["volume_pairs"]
        ],
        "distances": [
            dict(
                p,
                image=r["image"],
                split=r["split"],
                group=r["group"],
                cut=r["cut_references"][p["reference_id"]],
            )
            for r in rows
            for p in r["distance_pairs"]
        ],
    }
    for name, table in tables.items():
        with (destination / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(table[0]) if table else ["No observations"]
            )
            writer.writeheader()
            writer.writerows(table)
    lines = [
        "# Both-detector consensus with corrected cell channels",
        "",
        f"All {len(rows)} held-out scans evaluated with the current app models and a minimum roundness of {identity['minimum_roundness']:.2f}. Channels use declared Lam/LBR wavelengths, with ring evidence ranking multiple membrane stains. The nine scans without stain declarations were reviewed visually and use their sole 633 nm membrane metadata hint. Cell-channel indices in the CSV and JSON are zero-based.",
        "",
        "| Split | Scans | Matched complete cells (retained) | Volume mean absolute error | Distance MAE | Distance RMSE | Matched signals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, label in [
        ("test", "Test"),
        ("validation", "Validation"),
        ("all_heldout", "All held-out"),
    ]:
        s = summary[split]
        lines.append(
            f"| {label} | {s['scans']} | {s['volume']['n']} | {s['volume'].get('mape_percent', 0):.2f}% | {s['distance'].get('mae', 0):.3f} µm | {s['distance'].get('rmse', 0):.3f} µm | {s['distance']['n']} |"
        )
    lines.extend(
        [
            "",
            report["policy"],
            "",
            "Volume errors exclude manual cells cut by the image boundary. Distances use signals in matched complete cells. The JSON includes signed bias, medians, tails, and acquisition-bootstrap 95% intervals. Per-scan counts and individual measurement pairs are preserved in the CSV files. Neither model weights nor app defaults were changed.",
            "",
            "Reproduce with `scripts/benchmark_consensus_channels.py`; `--publish-only reports/consensus-correct-channels` verifies that every expected scan completed before publishing.",
        ]
    )
    test = summary["test"]
    if test["volume"]["n"] and test["distance"]["n"]:
        v, d = test["volume"], test["distance"]
        lines.extend(
            [
                "",
                f"{corrected} cell-channel assignments differ from the original audit. On test scans, mean signed volume error is {v['mean_signed_percent']:.2f}% (negative means volumes are too small). Mean signed distance error is {d['bias']:.3f} µm. Median absolute volume error is {v['median_ape_percent']:.2f}%; median absolute distance error is {d['median_absolute_error']:.3f} µm.",
            ]
        )
        if (
            "mape_95ci_acquisition_bootstrap" in v
            and "mae_95ci_acquisition_bootstrap" in d
        ):
            lines.extend(
                [
                    "",
                    f"Acquisition-bootstrap 95% intervals on the test set: volume MAPE {v['mape_95ci_acquisition_bootstrap'][0]:.2f}–{v['mape_95ci_acquisition_bootstrap'][1]:.2f}%; distance MAE {d['mae_95ci_acquisition_bootstrap'][0]:.3f}–{d['mae_95ci_acquisition_bootstrap'][1]:.3f} µm.",
                ]
            )
    (destination / "README.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "artifacts/refinement-fish/audit/audit.json",
    )
    parser.add_argument(
        "--channel-audit",
        type=Path,
        default=ROOT
        / "artifacts/preprocessed-stardist-stain-aware-20261010/audit/audit.json",
    )
    parser.add_argument(
        "--reuse", type=Path, default=ROOT / "artifacts/all-heldout-20261009-213558"
    )
    parser.add_argument("--roundness", type=float, default=0.72)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", choices=["gpu", "cpu", "auto"], default="gpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--publish-only", type=Path)
    args = parser.parse_args()
    if args.publish_only:
        report = publish(args.output, args.publish_only)
        print(json.dumps(report["summary"], indent=2))
        return 0
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid shard index")
    args.output.mkdir(parents=True, exist_ok=True)
    collection = json.loads(args.audit.read_text())
    models = frozen_models()
    old_identity = json.loads((args.reuse / "identity.json").read_text())
    for kind, model in models.items():
        if any(
            model[k] != old_identity["models"][kind][k]
            for k in ("manifest_sha256", "checkpoint_sha256")
        ):
            raise ValueError("Cached inference uses different frozen model: " + kind)
    cases = sorted(
        [
            c
            for c in collection["cases"]
            if c["audit"]["split"] in ("test", "validation")
        ],
        key=lambda c: (c["audit"]["split"] != "test", c["image"]),
    )
    training_groups = {
        c["audit"]["group"]
        for c in collection["cases"]
        if c["audit"]["split"] == "training"
    }
    if training_groups & {c["audit"]["group"] for c in cases}:
        raise ValueError("Cell training acquisition overlaps evaluation")
    signal_index = json.loads(
        Path(models["signals"]["manifest"]["data_index"]).read_text()
    )
    signal_training = {
        c["image"] for c in signal_index["cases"] if c["split"] == "training"
    }
    if signal_training & {c["image"] for c in cases}:
        raise ValueError("Signal training overlaps evaluation")
    for source in collection["sources"]:
        stat = Path(source["image"]).stat()
        if stat.st_size != source["size"] or stat.st_mtime_ns != source["mtime_ns"]:
            raise ValueError("Audited source image changed")
        for scene in source.get("scene_stats", []):
            stat = Path(scene["path"]).stat()
            if stat.st_size != scene["size"] or stat.st_mtime_ns != scene["mtime_ns"]:
                raise ValueError("Audited source annotations changed")
    channel_cases = {
        c["image"]: c for c in json.loads(args.channel_audit.read_text())["cases"]
    }
    channels = {c["image"]: channel_for(channel_cases[c["image"]]) for c in cases}
    identity = {
        "models": models,
        "audit_sha256": sha(args.audit),
        "channel_audit_sha256": sha(args.channel_audit),
        "minimum_roundness": args.roundness,
        "device": args.device,
        "channels": channels,
        "shards": args.shards,
    }
    identity = json.loads(json.dumps(identity))
    identity_path = args.output / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Run identity changed")
    else:
        save(identity_path, identity)
        save(
            args.output / "selection.json",
            {
                "cases": cases,
                "split_counts": dict(Counter(c["audit"]["split"] for c in cases)),
            },
        )
    if args.prepare_only:
        return 0
    cached = {}
    for path in args.reuse.glob("*/consensus/pipeline-result.json"):
        state = json.loads(path.read_text())
        cached[state["source"]] = state
    queue = [(i, c) for i, c in enumerate(cases) if i % args.shards == args.shard_index]
    app = QtCore.QCoreApplication([])
    controller = AnalysisController()
    rows, failures, active = [], [], {}
    started = time.monotonic()
    suffix = f"-shard-{args.shard_index}"

    def persist(status="running"):
        save(
            args.output / f"progress{suffix}.json",
            {
                "status": status,
                "completed": len(rows),
                "total": len(queue),
                "failures": failures,
                "elapsed_s": time.monotonic() - started,
                "active": active.get("image"),
                "stage": getattr(controller, "stage", None),
            },
        )

    def fail(message):
        failures.append({"image": active["image"], "error": message})
        persist("failed")
        print(message, flush=True)
        app.exit(1)

    def complete(state):
        try:
            save(active["folder"] / "pipeline-result.json", state)
            row = evaluate_run(active["case"], state, collection, active["folder"])
            row.update(
                mode="consensus",
                cut_references=active["case"]["audit"]["reference_cut"],
                cell_channel=channels[active["image"]][0],
                channel_policy=channels[active["image"]][1],
                reused_cells=state.get("reused_cells"),
            )
            save(active["folder"] / "metrics.json", row)
            rows.append(row)
            persist()
            print(
                "COMPLETE",
                len(rows),
                len(queue),
                row["scan"],
                row["matched_final"],
                flush=True,
            )
            QtCore.QTimer.singleShot(0, next_run)
        except Exception:  # noqa: BLE001 - record worker/evaluation failures before exiting Qt
            import traceback

            fail(traceback.format_exc())

    original_launch = controller._launch

    def launch(index, arguments):
        if controller.stages[index][0] == "export":
            controller.state.update(
                status="complete", benchmark_only=True, export_omitted=True
            )
            controller._stop()
            complete(
                dict(
                    controller.state,
                    report_path=str(controller.directory / "analysis-result.json"),
                )
            )
        else:
            original_launch(index, arguments)

    controller._launch = launch

    def next_run():
        if len(rows) == len(queue):
            app.quit()
            return
        i, case = queue[len(rows)]
        folder = args.output / (f"{i + 1:03d}-" + Path(case["image"]).stem)
        folder.mkdir(exist_ok=True)
        active.update(image=case["image"], case=case, folder=folder)
        if (folder / "metrics.json").exists():
            rows.append(json.loads((folder / "metrics.json").read_text()))
            QtCore.QTimer.singleShot(0, next_run)
            return
        if (folder / "pipeline-result.json").exists():
            complete(json.loads((folder / "pipeline-result.json").read_text()))
            return
        channel = channels[case["image"]][0]
        reuse = cached.get(case["image"])
        if reuse:
            path = Path(reuse["stages"]["consensus"]) / "report.json"
            report = json.loads(path.read_text())
            with np.load(report["files"][0]["predictions"], allow_pickle=False) as f:
                metadata = json.loads(str(f["metadata"]))
            if (
                metadata["channel"] != channel
                or metadata["fingerprint"] != case["audit"]["fingerprint"]
            ):
                reuse = None
        kwargs = {
            "device": args.device,
            "refinement": default_refinement_model(),
            "stardist_proposals": True,
            "consensus_model": default_proposal_model(),
            "roundness": args.roundness,
            "cell_channel": channel,
        }
        if reuse:
            controller._launch = lambda *_: None
            try:
                controller.start(
                    case["image"],
                    folder / "runs",
                    sensitive_stardist_model(),
                    default_models()[1],
                    **kwargs,
                )
            finally:
                controller._launch = launch
            controller.state["stages"].update(
                {
                    k: reuse["stages"][k]
                    for k in ("cells", "refine", "consensus_cells", "consensus")
                }
            )
            controller.state["reused_cells"] = reuse["report_path"]
            controller._persist()
            if args.roundness > 0:
                launch(
                    next(i for i, s in enumerate(controller.stages) if s[0] == "shape"),
                    [
                        "-m",
                        "fish_viewer.analysis.shape",
                        "--cells-report",
                        str(path),
                        "--output",
                        str(controller.directory / "work"),
                        "--minimum",
                        str(args.roundness),
                    ],
                )
            else:
                raise ValueError(
                    "Use a positive app roundness setting with cached runs"
                )
        else:
            controller.start(
                case["image"],
                folder / "runs",
                sensitive_stardist_model(),
                default_models()[1],
                **kwargs,
            )
        persist()

    controller.failed.connect(fail)
    controller.completed.connect(complete)
    QtCore.QTimer.singleShot(0, next_run)
    code = app.exec()
    if code or len(rows) != len(queue):
        return 1
    save(
        args.output / f"report{suffix}.json",
        {
            "status": "complete",
            "results": rows,
            "summary": summarize(rows),
            "identity": identity,
        },
    )
    persist("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
