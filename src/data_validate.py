#!/usr/bin/env python
"""Pre-training data validation gate for a YOLO detection dataset.

Exit code: 0 = passed, 1 = validation errors found, 2 = could not run.
Writes a JSON report (with a dataset fingerprint) and optionally logs to MLflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

SIGNATURES = {
    "jpeg": lambda b: b[:3] == b"\xff\xd8\xff",
    "png":  lambda b: b[:8] == b"\x89PNG\r\n\x1a\n",
    "bmp":  lambda b: b[:2] == b"BM",
    "webp": lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP",
}
EXT_TO_FORMAT = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
                 ".bmp": "bmp", ".webp": "webp"}


# ----------------------------------------------------------------- helpers
class Issues:
    """Collects error/warning counts plus a few examples per category."""

    def __init__(self, max_examples: int = 3):
        self.errors, self.warnings = Counter(), Counter()
        self.examples = defaultdict(list)
        self.max_examples = max_examples

    def add(self, level: str, category: str, msg: str = ""):
        bucket = self.errors if level == "error" else self.warnings
        bucket[category] += 1
        if msg and len(self.examples[category]) < self.max_examples:
            self.examples[category].append(msg)

    def to_dict(self):
        return {"errors": dict(self.errors), "warnings": dict(self.warnings),
                "examples": dict(self.examples)}


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["class_names"] = {int(k): v for k, v in cfg["class_names"].items()}
    return cfg


def scan_folder(folder: Path, allowed: set):
    files = [p for p in folder.iterdir() if p.is_file()]
    ok = sorted(p for p in files if p.suffix.lower() in allowed)
    skipped = sorted(p.name for p in files if p.suffix.lower() not in allowed)
    return ok, skipped


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------- check 1: pairing
def check_pairing(img_files, lbl_files, img_skipped, lbl_skipped, cfg, issues):
    img_stems = [p.stem for p in img_files]
    lbl_stems = [p.stem for p in lbl_files]
    img_set, lbl_set = set(img_stems), set(lbl_stems)

    for s in sorted(lbl_set - img_set):
        issues.add("error", "label_without_image", s)
    level = "warning" if cfg["allow_background_images"] else "error"
    for s in sorted(img_set - lbl_set):
        issues.add(level, "image_without_label", s)
    for name, stems in (("image", img_stems), ("label", lbl_stems)):
        for s, c in Counter(stems).items():
            if c > 1:
                issues.add("error", f"duplicate_{name}_stem", s)
    for n in img_skipped:
        issues.add("warning", "unexpected_file_in_images", n)
    for n in lbl_skipped:
        issues.add("warning", "unexpected_file_in_labels", n)


# ---------------------------------------------------- check 2: label format
def check_label_file(path: Path, cfg: dict):
    """Return (errors, warnings, class_ids, is_empty). Issues: (cat, line_no, detail)."""
    errors, warnings, class_ids, seen = [], [], [], set()
    n_classes, eps, min_box = len(cfg["class_names"]), cfg["eps"], cfg["min_box"]

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return [("unreadable_encoding", 0, "not valid UTF-8")], [], [], False

    is_empty = not text.strip()
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(("wrong_column_count", line_no, f"got {len(parts)}"))
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            errors.append(("non_numeric", line_no, line))
            continue
        if not all(math.isfinite(v) for v in values):
            errors.append(("nan_or_inf", line_no, line))
            continue

        cls_f, x, y, w, h = values
        ok = True
        if not cls_f.is_integer():
            errors.append(("class_not_integer", line_no, parts[0])); ok = False
        elif not 0 <= int(cls_f) < n_classes:
            errors.append(("class_out_of_range", line_no, parts[0])); ok = False
        if any(v < -eps or v > 1 + eps for v in (x, y, w, h)):
            errors.append(("not_normalized", line_no, line)); ok = False
        if w <= 0 or h <= 0:
            errors.append(("non_positive_size", line_no, line)); ok = False
        elif (x - w / 2 < -eps or x + w / 2 > 1 + eps or
              y - h / 2 < -eps or y + h / 2 > 1 + eps):
            errors.append(("box_outside_image", line_no, line)); ok = False

        if ok:
            if w < min_box or h < min_box:
                warnings.append(("tiny_box", line_no, line))
            key = (int(cls_f), round(x, 6), round(y, 6), round(w, 6), round(h, 6))
            if key in seen:
                warnings.append(("duplicate_box", line_no, line))
            seen.add(key)
            class_ids.append(int(cls_f))
    return errors, warnings, class_ids, is_empty


# -------------------------------------------------- check 3: image content
def check_image_file(path: Path, cfg: dict):
    """Return (errors, warnings, info). Issues are (category, detail)."""
    errors, warnings = [], []
    info = {"sha256": None, "width": None, "height": None}

    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return [("zero_bytes", "")], [], info

    raw = data.tobytes()
    header = raw[:16]
    real = next((n for n, t in SIGNATURES.items() if t(header)), None)
    if real is None:
        errors.append(("unknown_signature", header[:8].hex()))
    elif real != EXT_TO_FORMAT[path.suffix.lower()]:
        warnings.append(("extension_mismatch",
                         f"ext={path.suffix.lower()} content={real}"))

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        errors.append(("decode_failed", ""))
    else:
        h, w = img.shape[:2]
        info["width"], info["height"] = w, h
        if min(h, w) < cfg["min_image_side"]:
            warnings.append(("tiny_image", f"{w}x{h}"))

    if real == "jpeg" and raw[-2:] != b"\xff\xd9":
        warnings.append(("jpeg_missing_end_marker", ""))

    info["sha256"] = sha256_bytes(raw)
    return errors, warnings, info


# ------------------------------------------------------------ near-dups
def find_near_duplicates(file_index: dict, hamming_max: int):
    import imagehash
    from PIL import Image

    kept, rows = [], []
    for p, info in file_index.items():
        if not info["width"]:
            continue
        try:
            with Image.open(p) as im:
                rows.append(imagehash.phash(im).hash.flatten())
                kept.append(p)
        except Exception:
            continue
    if len(kept) < 2:
        return []
    H = np.array(rows)
    pairs = []
    for i in range(len(kept) - 1):
        d = (H[i] != H[i + 1:]).sum(axis=1)
        for off in np.nonzero(d <= hamming_max)[0]:
            j = i + 1 + int(off)
            if file_index[kept[i]]["sha256"] != file_index[kept[j]]["sha256"]:
                pairs.append((kept[i], kept[j], int(d[off])))
    return pairs


# ------------------------------------------------------------------ run
def run(cfg: dict) -> dict:
    root = Path(cfg["data_root"])
    exts = {e.lower() for e in cfg["image_exts"]}
    names = cfg["class_names"]
    splits_report, file_index, fp_entries = {}, {}, []

    for split in cfg["splits"]:
        issues = Issues()
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            issues.add("error", "missing_directory", f"{img_dir} or {lbl_dir}")
            splits_report[split] = {"n_images": 0, "n_label_files": 0, **issues.to_dict()}
            continue

        img_files, img_skipped = scan_folder(img_dir, exts)
        lbl_files, lbl_skipped = scan_folder(lbl_dir, {".txt"})
        check_pairing(img_files, lbl_files, img_skipped, lbl_skipped, cfg, issues)

        class_counts, n_empty, n_boxes = Counter(), 0, 0
        for f in lbl_files:
            errs, warns, cls_ids, is_empty = check_label_file(f, cfg)
            n_empty += is_empty
            for cat, ln, detail in errs:
                issues.add("error", cat, f"{f.name}:{ln} -> {detail}")
            for cat, ln, detail in warns:
                issues.add("warning", cat, f"{f.name}:{ln} -> {detail}")
            class_counts.update(cls_ids)
            n_boxes += len(cls_ids)
            fp_entries.append(f"{split}/labels/{f.name}:{sha256_bytes(f.read_bytes())}")

        sizes = Counter()
        for p in img_files:
            errs, warns, info = check_image_file(p, cfg)
            file_index[p] = {"split": split, **info}
            for cat, detail in errs:
                issues.add("error", cat, f"{p.name} -> {detail}")
            for cat, detail in warns:
                issues.add("warning", cat, f"{p.name} -> {detail}")
            if info["width"]:
                sizes[f"{info['width']}x{info['height']}"] += 1
            if info["sha256"]:
                fp_entries.append(f"{split}/images/{p.name}:{info['sha256']}")

        for cid in names:   # absent classes must be visible, not missing
            if class_counts.get(cid, 0) == 0:
                issues.add("warning", "class_absent_in_split", f"{cid}:{names[cid]}")

        splits_report[split] = {
            "n_images": len(img_files), "n_label_files": len(lbl_files),
            "n_empty_label_files": n_empty, "n_valid_boxes": n_boxes,
            "class_counts": {str(c): class_counts.get(c, 0) for c in names},
            "top_sizes": sizes.most_common(5),
            **issues.to_dict(),
        }

    # cross-split exact duplicates (leakage)
    g_issues = Issues(max_examples=10)
    by_hash = defaultdict(list)
    for p, info in file_index.items():
        if info["sha256"]:
            by_hash[info["sha256"]].append((info["split"], p.name))
    for group in (g for g in by_hash.values() if len(g) > 1):
        cross = len({s for s, _ in group}) > 1
        if cross:
            level = "error" if cfg["fail_on_cross_split_exact_duplicates"] else "warning"
            g_issues.add(level, "exact_duplicate_across_splits", str(group))
        else:
            g_issues.add("warning", "exact_duplicate_within_split", str(group))

    nd = cfg.get("near_duplicates", {})
    if nd.get("enabled"):
        try:
            pairs = find_near_duplicates(file_index, nd["hamming_max"])
            for a, b, d in pairs:
                sa, sb = file_index[a]["split"], file_index[b]["split"]
                cat = "near_duplicate_across_splits" if sa != sb else "near_duplicate_within_split"
                g_issues.add("warning", cat, f"d={d} {sa}/{a.name} <-> {sb}/{b.name}")
        except ImportError:
            g_issues.add("warning", "near_dup_skipped_imagehash_not_installed")

    fingerprint = sha256_bytes("\n".join(sorted(fp_entries)).encode("utf-8"))
    n_errors = sum(sum(r["errors"].values()) for r in splits_report.values()) \
        + sum(g_issues.errors.values())
    n_warnings = sum(sum(r["warnings"].values()) for r in splits_report.values()) \
        + sum(g_issues.warnings.values())

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    return {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {m: ver(m) for m in ("numpy", "cv2", "imagehash", "PIL", "yaml")},
        },
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
        "dataset_fingerprint": fingerprint,
        "splits": splits_report,
        "global": g_issues.to_dict(),
        "summary": {"n_errors": n_errors, "n_warnings": n_warnings,
                    "passed": n_errors == 0},
    }


# ------------------------------------------------------- output / logging
def print_summary(report: dict):
    for split, r in report["splits"].items():
        print(f"\n=== {split.upper()} === images: {r['n_images']} | "
              f"labels: {r['n_label_files']} | boxes: {r.get('n_valid_boxes', 0)}")
        print("errors:  ", r["errors"] or "none")
        print("warnings:", r["warnings"] or "none")
        for cat, ex in r["examples"].items():
            print(f"   {cat}: {ex}")
    g = report["global"]
    print("\n=== CROSS-SPLIT / DUPLICATES ===")
    print("errors:  ", g["errors"] or "none")
    print("warnings:", g["warnings"] or "none")
    for cat, ex in g["examples"].items():
        print(f"   {cat}: {ex[:3]}")
    s = report["summary"]
    print(f"\nfingerprint: {report['dataset_fingerprint'][:16]}...")
    print(f"RESULT: {'PASSED' if s['passed'] else 'FAILED'} "
          f"({s['n_errors']} errors, {s['n_warnings']} warnings)")


def log_to_mlflow(report: dict, report_path: Path, config_path: Path, args):
    import mlflow
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="data_validation"):
        mlflow.set_tag("stage", "data_validation")
        mlflow.set_tag("validation_passed", str(report["summary"]["passed"]))
        mlflow.log_param("dataset_fingerprint", report["dataset_fingerprint"])
        mlflow.log_metric("n_errors", report["summary"]["n_errors"])
        mlflow.log_metric("n_warnings", report["summary"]["n_warnings"])
        for split, r in report["splits"].items():
            mlflow.log_metric(f"{split}_n_images", r["n_images"])
            mlflow.log_metric(f"{split}_n_boxes", r.get("n_valid_boxes", 0))
            for cid, n in r.get("class_counts", {}).items():
                mlflow.log_metric(f"{split}_class_{cid}", n)
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(config_path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a YOLO dataset before training.")
    ap.add_argument("--config", default="validate_config.yaml")
    ap.add_argument("--mlflow", action="store_true", help="log results to MLflow")
    ap.add_argument("--tracking-uri", default=None,
                    help="MLflow tracking URI (default: local ./mlruns)")
    ap.add_argument("--experiment", default="ppe-data-validation")
    args = ap.parse_args(argv)

    try:
        config_path = Path(args.config)
        cfg = load_config(config_path)
        report = run(cfg)
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"Could not run validation: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"validation_{stamp}_{report['dataset_fingerprint'][:8]}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report)
    print(f"report written to: {report_path}")

    if args.mlflow:
        try:
            log_to_mlflow(report, report_path, config_path, args)
            print("logged to MLflow")
        except ImportError:
            print("mlflow not installed; skipping logging", file=sys.stderr)

    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())