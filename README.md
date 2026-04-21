# MaskLens

Desktop tool for auditing SAM3 segmentation masks against ground-truth annotations in pothole detection datasets.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue) ![PySide6](https://img.shields.io/badge/UI-PySide6-green)

---

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py                  # pick dataset via UI
python app.py /path/to/dataset # load directly
```

---

## Dataset structure

```
dataset/
├── images/              # source images  (.jpg .jpeg .png .bmp .tiff .webp)
├── bbox/                # YOLO bounding boxes  (one .txt per image)
├── masks/               # GT binary masks  (optional)
├── sam_masks/           # SAM3 masks  (named <stem>_SAM3.png)
└── sam_masks_cleaned/   # edited masks — written automatically on approval
```

### Bounding box format (YOLO)

One box per line — `class cx cy w h` with coordinates normalised to [0, 1]:

```
0 0.545833 0.960000 0.098333 0.080000
0 0.701667 0.719167 0.106667 0.048333
```

### Mask format

Binary PNG: object pixels = 255 (white), background = 0 (black).

---

## Multi-user workflow

When a dataset is loaded for the first time, MaskLens asks for every auditor's name and shuffles the images into equal partitions. This creates `assignment.csv` in the dataset root — **delete it to re-partition**.

```
dataset/
├── assignment.csv            # image → user mapping  (auto-generated)
├── audit_results_Alice.csv   # Alice's decisions
└── audit_results_Bob.csv     # Bob's decisions
```

Each subsequent session shows a user-select screen with live progress per person. Use **Switch** in the header to hand off to a co-auditor mid-session. Both users can run the app simultaneously on the same folder without conflicts.

---

## Controls

| Key | Action |
|-----|--------|
| `↑` | **Segmentation** — approve SAM3 mask (original or cleaned) |
| `↓` | **Trash** — mask *and* original annotation unusable |
| `→` | **Detection** — mask disapproved, original annotation is good |
| `←` | **Return later** — re-queued at end of session |
| `⌫ Backspace` | **Undo** — step back and erase the previous decision |

Progress is saved after every keystroke. Closing mid-session resumes exactly where you left off.

---

## Mask editor

Press **✏ Edit Mask** in the controls bar to enter edit mode. The annotated panel gains an orange border and a crosshair cursor.

- **Click** on any white region to erase that entire connected component instantly.
- **↺ Reset** reverts all erases for the current image back to the original mask.
- Edit mode stays active as you move between images — toggle it off when done.
- Pressing `↑` while edits are present saves the cleaned mask to `sam_masks_cleaned/` and records `segmentation_cleaned` in the CSV instead of `segmentation_original`.

The editor is disabled for images that have no SAM3 mask.

---

## Output

```csv
image_name,decision
img001.jpg,segmentation_original
img002.jpg,segmentation_cleaned
img003.jpg,detection
img004.jpg,nothing
```

| Decision | Meaning |
|----------|---------|
| `segmentation_original` | SAM3 mask approved as-is |
| `segmentation_cleaned` | SAM3 mask approved after manual erasing — cleaned mask saved to `sam_masks_cleaned/` |
| `detection` | Discard SAM3 mask, keep original bbox as GT |
| `nothing` | Discard everything — sample is unusable |
| `return_later` | Deferred — will reappear at end of queue |
