"""
Populate RDD022_auditado/ from the audit CSVs.

Decision rules
--------------
segmentation_original → images/, sam_masks/{stem}_SAM3.png, bboxes/{stem}.txt (generated)
segmentation_cleaned  → images/, sam_masks/{stem}.png,      bboxes/{stem}.txt (generated)
detection             → images/, bboxes/{stem}.txt (original ground-truth)
nothing               → skip
"""

import csv
import shutil
from pathlib import Path

BASE        = Path(__file__).parent
SRC_IMAGES  = BASE / "images"
SRC_MASKS   = BASE / "sam_masks"
SRC_CLEAN   = BASE / "sam_masks_cleaned"
SRC_BBOXES  = BASE / "bboxes"           # original ground-truth
SRC_GEN     = BASE / "bboxes_generated" # generated from masks

OUT         = BASE / "RDD022_auditado"
OUT_IMAGES  = OUT / "images"
OUT_MASKS   = OUT / "sam_masks"
OUT_BBOXES  = OUT / "bboxes"

CAIO_CSV     = BASE / "audit_results_Caio.csv"
FERNANDO_CSV = BASE / "audit_results_Fernando.csv"

VALID = {"segmentation_original", "segmentation_cleaned", "detection"}


def load_decisions() -> dict:
    """Return {stem: decision}. Caio takes precedence on conflicts."""
    decisions = {}

    with open(CAIO_CSV, newline="") as f:
        for row in csv.DictReader(f):
            d = row["decision"].strip()
            if d in VALID:
                decisions[Path(row["image_name"]).stem] = d

    with open(FERNANDO_CSV, newline="") as f:
        for row in csv.DictReader(f):
            d = row["decision"].strip()
            if d in VALID:
                decisions[Path(row["image_name"]).stem] = d
                
    return decisions


def copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    decisions = load_decisions()
    print(f"Decisions loaded: {len(decisions)}")

    counts = {"images": 0, "masks": 0, "bboxes": 0, "missing": 0}

    for stem, decision in sorted(decisions.items()):
        img_src = SRC_IMAGES / f"{stem}.jpg"

        # ── image ────────────────────────────────────────────────────────────
        if copy(img_src, OUT_IMAGES / f"{stem}.jpg"):
            counts["images"] += 1
        else:
            print(f"  [MISSING image] {stem}")
            counts["missing"] += 1

        # ── mask + generated bbox ────────────────────────────────────────────
        if decision == "segmentation_original":
            mask_src = SRC_MASKS / f"{stem}_SAM3.png"
            mask_dst = OUT_MASKS / f"{stem}_SAM3.png"
            bbox_src = SRC_GEN / f"{stem}.txt"

        elif decision == "segmentation_cleaned":
            mask_src = SRC_CLEAN / f"{stem}.png"
            mask_dst = OUT_MASKS / f"{stem}.png"
            bbox_src = SRC_GEN / f"{stem}.txt"

        else:  # detection
            mask_src = None
            bbox_src = SRC_BBOXES / f"{stem}.txt"

        if mask_src is not None:
            if copy(mask_src, mask_dst):
                counts["masks"] += 1
            else:
                print(f"  [MISSING mask] {stem} ({decision})")
                counts["missing"] += 1

        bbox_dst = OUT_BBOXES / f"{stem}.txt"
        if bbox_src.exists() and bbox_src.stat().st_size > 0:
            copy(bbox_src, bbox_dst)
            counts["bboxes"] += 1
        else:
            if decision == "detection":
                print(f"  [MISSING bbox] {stem}")
                counts["missing"] += 1
            # empty generated bbox files are silently skipped

    print(f"\nDone.")
    print(f"  images : {counts['images']}")
    print(f"  masks  : {counts['masks']}")
    print(f"  bboxes : {counts['bboxes']}")
    print(f"  missing: {counts['missing']}")


if __name__ == "__main__":
    main()
