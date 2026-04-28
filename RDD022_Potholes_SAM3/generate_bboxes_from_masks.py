"""
Generate YOLO bounding boxes from SAM3 segmentation masks.

Reads audit_results_Caio.csv and audit_results_Fernando.csv to find images
marked as segmentation_cleaned or segmentation_original, loads the corresponding
mask, extracts one bbox per connected component, and writes YOLO-format .txt
files to bboxes_generated/.

YOLO format: <class_id> <cx> <cy> <w> <h>  (all normalized to [0, 1])
class_id is always 0 (pothole).
"""

import csv
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent

SAM_MASKS_DIR = BASE_DIR / "sam_masks"
SAM_MASKS_CLEANED_DIR = BASE_DIR / "sam_masks_cleaned"
OUTPUT_DIR = BASE_DIR / "bboxes_generated"
CAIO_CSV     = BASE_DIR / "audit_results_Caio.csv"
FERNANDO_CSV = BASE_DIR / "audit_results_Fernando.csv"

# Minimum pixel area to keep a connected component (filters out noise)
MIN_AREA = 100


def load_audit_decisions() -> dict:
    """Return {stem: decision} for all images to process.

    Both CSVs contribute segmentation_original and segmentation_cleaned.
    Caio takes precedence when the same image appears in both.
    Missing mask files are handled gracefully downstream.
    """
    decisions: dict = {}

    with open(CAIO_CSV, newline="") as f:
        for row in csv.DictReader(f):
            d = row["decision"].strip()
            if d in ("segmentation_original", "segmentation_cleaned"):
                decisions[Path(row["image_name"].strip()).stem] = d

    with open(FERNANDO_CSV, newline="") as f:
        for row in csv.DictReader(f):
            d = row["decision"].strip()
            if d in ("segmentation_original", "segmentation_cleaned"):
                stem = Path(row["image_name"].strip()).stem
                if stem not in decisions:          # Caio takes precedence
                    decisions[stem] = d

    return decisions


def find_mask_path(stem: str, decision: str) -> Optional[Path]:
    """Locate the mask file for a given image stem and audit decision."""
    if decision == "segmentation_original":
        candidate = SAM_MASKS_DIR / f"{stem}_SAM3.png"
        if candidate.exists():
            return candidate
        # Some files in sam_masks don't have the _SAM3 suffix
        candidate_no_suffix = SAM_MASKS_DIR / f"{stem}.png"
        if candidate_no_suffix.exists():
            return candidate_no_suffix
    elif decision == "segmentation_cleaned":
        candidate = SAM_MASKS_CLEANED_DIR / f"{stem}.png"
        if candidate.exists():
            return candidate
    return None


def mask_to_yolo_bboxes(mask_path: Path) -> list[str]:
    """Return YOLO bbox strings for each connected component in the mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    h, w = mask.shape
    binary = (mask > 127).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    lines = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]

        cx = (x + comp_w / 2) / w
        cy = (y + comp_h / 2) / h
        w_norm = comp_w / w
        h_norm = comp_h / h

        lines.append(f"0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")

    return lines


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    decisions = load_audit_decisions()
    print(f"Found {len(decisions)} images to process.")

    skipped, generated, empty = 0, 0, 0

    for stem, decision in sorted(decisions.items()):
        mask_path = find_mask_path(stem, decision)
        if mask_path is None:
            print(f"  [SKIP] mask not found for {stem} ({decision})")
            skipped += 1
            continue

        try:
            lines = mask_to_yolo_bboxes(mask_path)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            skipped += 1
            continue

        out_path = OUTPUT_DIR / f"{stem}.txt"
        if lines:
            out_path.write_text("\n".join(lines) + "\n")
            generated += 1
        else:
            # Write empty file to indicate the image was processed but had no valid mask regions
            out_path.write_text("")
            empty += 1
            print(f"  [EMPTY] no valid components in {stem}")

    print(f"\nDone. Generated: {generated}, empty: {empty}, skipped: {skipped}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
