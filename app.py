#!/usr/bin/env python3
"""
MaskLens – Audit SAM3 segmentation masks for pothole detection datasets.

Usage:
    python app.py [dataset_folder]

Controls:
    ↑   Segmentation – SAM3 mask approved
    ↓   Trash – mask disapproved AND original annotation is bad (unusable)
    →   Detection – mask disapproved but original annotation is good (keep as GT)
    ←   Return later (re-queued at end)

Bbox file format:
    YOLO format – one bounding box per line: class cx cy w h
    (class index + center x/y and width/height, all normalized 0-1).
    Multiple boxes per file are supported.
"""

import sys
import csv
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSlider, QFileDialog, QColorDialog,
    QProgressBar, QSizePolicy, QFrame, QMessageBox,
    QDialog, QLineEdit, QDialogButtonBox,
)
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QKeySequence, QShortcut


# ──────────────────────────────────────────────────────── image helpers ───────

def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    # .copy() pins the memory so Qt doesn't read freed data
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def load_bboxes(path: Path, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """Parse YOLO bbox file: 'class cx cy w h' per line (normalized 0-1), returns absolute pixel coords."""
    boxes: list[tuple[int, int, int, int]] = []
    try:
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = int((cx - bw / 2) * img_w)
                        y1 = int((cy - bh / 2) * img_h)
                        x2 = int((cx + bw / 2) * img_w)
                        y2 = int((cy + bh / 2) * img_h)
                        boxes.append((x1, y1, x2, y2))
                    except ValueError:
                        pass
    except OSError:
        pass
    return boxes


def erase_component(mask: np.ndarray, px: int, py: int) -> Optional[np.ndarray]:
    """Zero out the connected component at (px, py). Returns None if pixel is already background."""
    if not (0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]):
        return None
    if mask[py, px] == 0:
        return None
    _, labels = cv2.connectedComponents(mask)
    result = mask.copy()
    result[labels == labels[py, px]] = 0
    return result


def compose_overlay(
    original: np.ndarray,
    gt_mask: Optional[np.ndarray],
    bboxes: list[tuple[int, int, int, int]],
    sam_mask: Optional[np.ndarray],
    gt_bgr: tuple[int, int, int],
    sam_bgr: tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    """
    Blend GT annotations and SAM3 mask on top of the original image.
    Order: GT mask fill → GT bbox fill → SAM3 mask fill → GT bbox outline.
    """
    h, w = original.shape[:2]
    canvas = original.astype(np.float32)

    def _blend(region: np.ndarray, color_bgr: tuple[int, int, int]):
        nonlocal canvas
        color = np.array(color_bgr, dtype=np.float32)
        mask3 = region[:, :, np.newaxis]
        canvas = np.where(mask3, canvas * (1.0 - opacity) + color * opacity, canvas)

    # GT mask
    if gt_mask is not None:
        m = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        _blend(m > 127, gt_bgr)

    # GT bboxes (filled, same color as GT mask)
    if bboxes:
        bbox_layer = np.zeros((h, w), dtype=bool)
        for x1, y1, x2, y2 in bboxes:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c > x1c and y2c > y1c:
                bbox_layer[y1c:y2c, x1c:x2c] = True
        _blend(bbox_layer, gt_bgr)

    # SAM3 mask
    if sam_mask is not None:
        m = cv2.resize(sam_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        _blend(m > 127, sam_bgr)

    out = np.clip(canvas, 0, 255).astype(np.uint8)

    # Solid bbox outline on top for precise boundary visibility
    if bboxes:
        for x1, y1, x2, y2 in bboxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), gt_bgr, 2)

    return out


# ─────────────────────────────────────────────────────────── widget ───────────

class ScaledImageLabel(QLabel):
    """QLabel that scales its pixmap to fit while preserving aspect ratio.
    In edit mode it captures left-clicks and emits image-space coordinates."""

    # Emits (image_x, image_y) in the coordinate space of the full-resolution pixmap
    image_clicked = Signal(int, int)

    _STYLE_NORMAL = "background-color: #1a1a1a; border: 1px solid #3a3a3a;"
    _STYLE_EDIT   = "background-color: #1a1a1a; border: 2px solid #ff9800;"

    def __init__(self, placeholder: str = "No image"):
        super().__init__()
        self._pixmap: Optional[QPixmap] = None
        self._placeholder = placeholder
        self._edit_mode = False
        self._render_rect = QRect()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(self._STYLE_NORMAL)
        self.setText(self._placeholder)

    def set_edit_mode(self, active: bool):
        self._edit_mode = active
        if active:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setStyleSheet(self._STYLE_EDIT)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setStyleSheet(self._STYLE_NORMAL)

    def set_image(self, pixmap: Optional[QPixmap]):
        self._pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            super().setPixmap(QPixmap())
            self.setText(self._placeholder)
        else:
            self.setText("")
            self._refresh()

    def _refresh(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(scaled)
            x_off = (self.width()  - scaled.width())  // 2
            y_off = (self.height() - scaled.height()) // 2
            self._render_rect = QRect(x_off, y_off, scaled.width(), scaled.height())

    def mousePressEvent(self, event):
        if (self._edit_mode
                and event.button() == Qt.MouseButton.LeftButton
                and self._pixmap and not self._pixmap.isNull()
                and self._render_rect.width() > 0):
            r = self._render_rect
            cx = event.position().x() - r.x()
            cy = event.position().y() - r.y()
            if 0 <= cx < r.width() and 0 <= cy < r.height():
                img_x = int(cx / r.width()  * self._pixmap.width())
                img_y = int(cy / r.height() * self._pixmap.height())
                self.image_clicked.emit(img_x, img_y)
                return
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


# ──────────────────────────────────────────── user assignment helpers ─────────

_ASSIGNMENT_FILE = "assignment.csv"


def _partition(items: list, n: int) -> list[list]:
    """Shuffle items and split into n roughly-equal groups (round-robin)."""
    pool = items[:]
    random.shuffle(pool)
    groups: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(pool):
        groups[i % n].append(item)
    return groups


class SetupUsersDialog(QDialog):
    """First-run dialog: collect user names and generate assignment.csv."""

    def __init__(self, image_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup Users")
        self.setModal(True)
        self.setMinimumWidth(380)

        self._edits: list[QLineEdit] = []

        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)
        vbox.setContentsMargins(20, 16, 20, 16)

        info = QLabel(
            f"No user assignment found for this dataset.\n"
            f"{image_count} images will be shuffled and split evenly.\n\n"
            f"Enter the name of every auditor:"
        )
        info.setWordWrap(True)
        vbox.addWidget(info)

        self._rows = QVBoxLayout()
        vbox.addLayout(self._rows)
        self._add_row("User 1")
        self._add_row("User 2")

        add_btn = QPushButton("+ Add user")
        add_btn.clicked.connect(lambda: self._add_row())
        vbox.addWidget(add_btn)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Create assignment")
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        vbox.addWidget(btns)

    def _add_row(self, placeholder: str = ""):
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder or f"User {len(self._edits) + 1}")
        self._edits.append(edit)
        self._rows.addWidget(edit)

    def _on_ok(self):
        names = self.user_names
        if not names:
            QMessageBox.warning(self, "Error", "Enter at least one user name.")
            return
        if len(names) != len(set(names)):
            QMessageBox.warning(self, "Error", "User names must be unique.")
            return
        self.accept()

    @property
    def user_names(self) -> list[str]:
        return [e.text().strip() for e in self._edits if e.text().strip()]


class SelectUserDialog(QDialog):
    """Session-start dialog: pick which user is currently working."""

    def __init__(self, users: list[str], counts: dict[str, tuple[int, int]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Who is working?")
        self.setModal(True)
        self.setMinimumWidth(300)

        self._selected: Optional[str] = None

        vbox = QVBoxLayout(self)
        vbox.setSpacing(8)
        vbox.setContentsMargins(20, 16, 20, 16)
        vbox.addWidget(QLabel("Select your name to continue:"))

        for user in users:
            done, total = counts.get(user, (0, 0))
            pct = f"{round(done/total*100)}%" if total else "0%"
            btn = QPushButton(f"{user}   ({done}/{total}  {pct})")
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda _, u=user: self._pick(u))
            vbox.addWidget(btn)

    def _pick(self, user: str):
        self._selected = user
        self.accept()

    @property
    def selected_user(self) -> Optional[str]:
        return self._selected


# ────────────────────────────────────────────────────────── data layer ────────

# Terminal decisions — anything not in this set is still pending
_FINALIZED = {"segmentation", "segmentation_original", "segmentation_cleaned", "nothing", "detection"}


class Dataset:
    def __init__(self, root: str, user: str):
        self.root = Path(root)
        self.user = user
        self.csv_path = self.root / f"audit_results_{user}.csv"

        assignment = Dataset.load_assignment(self.root)
        all_scanned = Dataset.scan_images(self.root)
        if assignment:
            self.all_images = [img for img in all_scanned if assignment.get(img) == user]
        else:
            self.all_images = all_scanned

        self.decisions: dict[str, str] = self._load_csv()

        finalized = {k for k, v in self.decisions.items() if v in _FINALIZED}
        return_later = [img for img in self.all_images if self.decisions.get(img) == "return_later"]
        pending = [img for img in self.all_images if img not in finalized and img not in return_later]

        # Pending first, then images flagged "return later" from previous sessions
        self.queue: list[str] = pending + return_later
        self.pos: int = 0
        # Each entry: (image_name, decision_before_action, pos_before_action)
        self._history: list[tuple[str, Optional[str], int]] = []

    # ── scanning & assignment ──────────────────────────────────────────────────

    @staticmethod
    def scan_images(root: Path) -> list[str]:
        d = root / "images"
        if not d.exists():
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        return sorted(f.name for f in d.iterdir() if f.suffix.lower() in exts)

    @staticmethod
    def load_assignment(root: Path) -> Optional[dict[str, str]]:
        path = root / _ASSIGNMENT_FILE
        if not path.exists():
            return None
        out: dict[str, str] = {}
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if "image_name" in row and "user" in row:
                        out[row["image_name"]] = row["user"]
        except Exception:
            return None
        return out or None

    @staticmethod
    def create_assignment(root: Path, all_images: list[str], users: list[str]):
        groups = _partition(all_images, len(users))
        mapping: dict[str, str] = {}
        for user, group in zip(users, groups):
            for img in group:
                mapping[img] = user
        with open(root / _ASSIGNMENT_FILE, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image_name", "user"])
            for img in all_images:
                w.writerow([img, mapping[img]])

    @staticmethod
    def progress_for_all_users(root: Path) -> dict[str, tuple[int, int]]:
        """Return {user: (done, total)} for every user in the assignment."""
        assignment = Dataset.load_assignment(root)
        if not assignment:
            return {}
        totals: dict[str, int] = {}
        for user in assignment.values():
            totals[user] = totals.get(user, 0) + 1
        done_counts: dict[str, int] = {u: 0 for u in totals}
        for user in totals:
            csv_path = root / f"audit_results_{user}.csv"
            if csv_path.exists():
                try:
                    with open(csv_path, newline="") as fh:
                        for row in csv.DictReader(fh):
                            if row.get("decision") in _FINALIZED:
                                done_counts[user] = done_counts.get(user, 0) + 1
                except Exception:
                    pass
        return {u: (done_counts[u], totals[u]) for u in totals}

    def _load_csv(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.csv_path.exists():
            try:
                with open(self.csv_path, newline="") as fh:
                    for row in csv.DictReader(fh):
                        if "image_name" in row and "decision" in row:
                            out[row["image_name"]] = row["decision"]
            except Exception:
                pass
        return out

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self):
        try:
            with open(self.csv_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["image_name", "decision"])
                w.writerows(self.decisions.items())
        except OSError as e:
            print(f"[MaskLens] could not save: {e}")

    # ── navigation state ──────────────────────────────────────────────────────

    @property
    def current(self) -> Optional[str]:
        return self.queue[self.pos] if 0 <= self.pos < len(self.queue) else None

    def decide(self, verdict: str) -> bool:
        img = self.current
        if img is None:
            return False
        self._history.append((img, self.decisions.get(img), self.pos))
        self.decisions[img] = verdict
        if verdict == "return_later":
            self.queue.pop(self.pos)
            self.queue.append(img)
            # pos stays the same – the next item slides into the current slot
        else:
            self.pos += 1
        self.save()
        return True

    def back(self) -> bool:
        if not self._history:
            return False
        img, prev_decision, old_pos = self._history.pop()
        # Remove img from wherever it currently sits in the queue
        if img in self.queue:
            self.queue.remove(img)
        # Re-insert at the position it had before the action
        insert_at = min(old_pos, len(self.queue))
        self.queue.insert(insert_at, img)
        self.pos = insert_at
        if prev_decision is None:
            self.decisions.pop(img, None)
        else:
            self.decisions[img] = prev_decision
        self.save()
        return True

    @property
    def progress(self) -> tuple[int, int]:
        done = sum(1 for v in self.decisions.values() if v in _FINALIZED)
        return done, len(self.all_images)

    def save_cleaned_mask(self, image_name: str, mask: np.ndarray):
        folder = self.root / "sam_masks_cleaned"
        folder.mkdir(exist_ok=True)
        stem = Path(image_name).stem
        cv2.imwrite(str(folder / f"{stem}.png"), mask)

    # ── path resolution ───────────────────────────────────────────────────────

    def get_paths(self, name: str) -> dict[str, Optional[Path]]:
        stem = Path(name).stem
        paths: dict[str, Optional[Path]] = {
            "image": self.root / "images" / name,
            "mask": None,
            "bbox": None,
            "sam_mask": None,
        }
        lookups = [
            ("masks",      "mask",     [f"{stem}_mask.png",     f"{stem}.png",     f"{stem}_mask.jpg"]),
            ("bbox",       "bbox",     [f"{stem}.txt",          f"{stem}_bbox.txt"]),
            ("bboxes",     "bbox",     [f"{stem}_bbox.txt",     f"{stem}.txt"]),
            ("sam_masks",  "sam_mask", [f"{stem}_SAM3.png",     f"{stem}_sam_mask.png", f"{stem}.png", f"{stem}_sam_mask.jpg"]),
            ("sam3_masks", "sam_mask", [f"{stem}_SAM3.png",     f"{stem}_sam_mask.png", f"{stem}.png", f"{stem}_sam_mask.jpg"]),
        ]
        for folder, key, variants in lookups:
            if paths[key] is not None:   # already found by a previous rule
                continue
            d = self.root / folder
            if d.exists():
                for v in variants:
                    fp = d / v
                    if fp.exists():
                        paths[key] = fp
                        break
        return paths


# ──────────────────────────────────────────────────────── main window ─────────

_DARK   = "#2b2b2b"
_TEXT   = "#dcdcdc"
_MUTED  = "#888888"
_BORDER = "#3a3a3a"
_ACCENT = "#5a9fd4"

STYLESHEET = f"""
QWidget      {{ background-color: {_DARK}; color: {_TEXT}; font-size: 13px; }}
QLabel       {{ color: {_TEXT}; }}
QPushButton  {{
    background-color: #383838; color: {_TEXT};
    border: 1px solid #555; padding: 4px 12px; border-radius: 4px;
}}
QPushButton:hover {{ background-color: #484848; }}
QSlider::groove:horizontal {{
    height: 6px; background: #444; border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: #aaaaaa; width: 14px; height: 14px;
    margin: -4px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {_ACCENT}; border-radius: 3px; }}
QProgressBar {{
    border: 1px solid #555; border-radius: 4px;
    text-align: center; background-color: #383838; color: {_TEXT};
}}
QProgressBar::chunk {{ background-color: {_ACCENT}; border-radius: 3px; }}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset: Optional[Dataset] = None

        # Currently loaded image data
        self._orig_img:  Optional[np.ndarray] = None
        self._gt_mask:   Optional[np.ndarray] = None
        self._bboxes:    list[tuple[int, int, int, int]] = []
        self._sam_mask:  Optional[np.ndarray] = None

        # Overlay configuration
        self.gt_color  = QColor("#00e676")   # bright green
        self.sam_color = QColor("#ff5252")   # bright red
        self.opacity   = 0.45

        # Edit-mode state (reset per image, mode toggle is sticky)
        self._edit_mode: bool = False
        self._sam_mask_edited: Optional[np.ndarray] = None
        self._mask_was_edited: bool = False

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("MaskLens")
        self.setMinimumSize(980, 640)
        self.setStyleSheet(STYLESHEET)

        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 0, 0)

        vbox.addWidget(self._build_header())
        vbox.addWidget(self._hsep())
        vbox.addWidget(self._build_progress_row())
        vbox.addWidget(self._hsep())
        vbox.addWidget(self._build_image_panels(), stretch=1)
        vbox.addWidget(self._hsep())
        vbox.addWidget(self._build_controls_row())
        vbox.addWidget(self._hsep())
        vbox.addWidget(self._build_shortcuts_row())

        self._refresh_progress()
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Up),        self).activated.connect(self._cmd_segmentation)
        QShortcut(QKeySequence(Qt.Key.Key_Down),      self).activated.connect(self._cmd_nothing)
        QShortcut(QKeySequence(Qt.Key.Key_Right),     self).activated.connect(self._cmd_detection)
        QShortcut(QKeySequence(Qt.Key.Key_Left),      self).activated.connect(self._cmd_return_later)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self).activated.connect(self._cmd_back)

    def _hsep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {_BORDER}; border: none;")
        return sep

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(46)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(14, 0, 14, 0)

        title = QLabel("MaskLens")
        title.setFont(QFont("", 15, QFont.Weight.Bold))

        divider = QLabel("│")
        divider.setStyleSheet(f"color: {_BORDER};")

        self._ds_label = QLabel("No dataset loaded — press Change to select a folder")
        self._ds_label.setStyleSheet(f"color: {_MUTED};")

        self._user_label = QLabel("")
        self._user_label.setStyleSheet(
            f"color: {_ACCENT}; font-weight: bold; font-size: 13px;"
        )

        self._switch_btn = QPushButton("Switch")
        self._switch_btn.setFixedWidth(64)
        self._switch_btn.setVisible(False)
        self._switch_btn.clicked.connect(self._switch_user)

        change_btn = QPushButton("Change")
        change_btn.setFixedWidth(76)
        change_btn.clicked.connect(self._pick_dataset)

        lay.addWidget(title)
        lay.addWidget(divider)
        lay.addWidget(self._ds_label, stretch=1)
        lay.addSpacing(12)
        lay.addWidget(self._user_label)
        lay.addSpacing(6)
        lay.addWidget(self._switch_btn)
        lay.addSpacing(6)
        lay.addWidget(change_btn)
        return w

    def _build_progress_row(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(38)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(14, 5, 14, 5)

        self._prog_label = QLabel("Progress: 0 / 0")
        self._prog_label.setFixedWidth(148)

        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setFixedHeight(16)
        self._prog_bar.setTextVisible(False)

        self._pct_label = QLabel("0%")
        self._pct_label.setFixedWidth(36)

        self._img_name_label = QLabel("")
        self._img_name_label.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        self._img_name_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        lay.addWidget(self._prog_label)
        lay.addWidget(self._prog_bar, stretch=1)
        lay.addWidget(self._pct_label)
        lay.addSpacing(20)
        lay.addWidget(self._img_name_label, stretch=1)
        return w

    def _build_image_panels(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(12)

        def _panel(title: str) -> tuple[QWidget, ScaledImageLabel]:
            container = QWidget()
            vl = QVBoxLayout(container)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(4)
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
            img_lbl = ScaledImageLabel("No image")
            vl.addWidget(lbl)
            vl.addWidget(img_lbl)
            return container, img_lbl

        orig_w, self._orig_panel = _panel("Original")
        ann_w,  self._ann_panel  = _panel("Annotated  (GT + SAM3 overlay)")
        self._ann_panel.image_clicked.connect(self._on_mask_click)
        lay.addWidget(orig_w)
        lay.addWidget(ann_w)
        return w

    def _build_controls_row(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(46)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(10)

        self._gt_btn = self._make_color_btn(self.gt_color)
        self._gt_btn.clicked.connect(lambda: self._pick_color("gt"))

        self._sam_btn = self._make_color_btn(self.sam_color)
        self._sam_btn.clicked.connect(lambda: self._pick_color("sam"))

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(int(self.opacity * 100))
        self._opacity_slider.setFixedWidth(160)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)

        self._opacity_lbl = QLabel(f"{int(self.opacity * 100)}%")
        self._opacity_lbl.setFixedWidth(36)

        self._edit_btn = QPushButton("✏  Edit Mask")
        self._edit_btn.setCheckable(True)
        self._edit_btn.setFixedHeight(30)
        self._edit_btn.setEnabled(False)
        self._edit_btn.toggled.connect(self._toggle_edit_mode)

        self._reset_btn = QPushButton("↺  Reset")
        self._reset_btn.setFixedHeight(30)
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self._reset_edit_state)

        lay.addWidget(QLabel("GT color:"))
        lay.addWidget(self._gt_btn)
        lay.addSpacing(16)
        lay.addWidget(QLabel("SAM3 color:"))
        lay.addWidget(self._sam_btn)
        lay.addSpacing(20)
        lay.addWidget(QLabel("Opacity:"))
        lay.addWidget(self._opacity_slider)
        lay.addWidget(self._opacity_lbl)
        lay.addStretch()
        lay.addWidget(self._edit_btn)
        lay.addSpacing(6)
        lay.addWidget(self._reset_btn)
        return w

    def _build_shortcuts_row(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(34)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(14, 0, 14, 0)
        lbl = QLabel("↑  Segmentation      ↓  Trash      →  Detection      ←  Return later      ⌫  Undo")
        lbl.setStyleSheet(f"color: {_MUTED}; font-size: 12px;")

        self._undo_btn = QPushButton("↩  Undo")
        self._undo_btn.setFixedSize(80, 24)
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._cmd_back)

        lay.addWidget(lbl)
        lay.addStretch()
        lay.addWidget(self._undo_btn)
        lay.addSpacing(4)
        return w

    # ── color button helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_color_btn(color: QColor) -> QPushButton:
        btn = QPushButton()
        btn.setFixedSize(28, 28)
        MainWindow._apply_color_style(btn, color)
        return btn

    @staticmethod
    def _apply_color_style(btn: QPushButton, color: QColor):
        btn.setStyleSheet(
            f"background-color: {color.name()}; "
            f"border: 2px solid #666; border-radius: 4px;"
        )

    def _qcolor_to_bgr(self, c: QColor) -> tuple[int, int, int]:
        return c.blue(), c.green(), c.red()

    # ── slots ─────────────────────────────────────────────────────────────────

    def _pick_color(self, which: str):
        current = self.gt_color if which == "gt" else self.sam_color
        color = QColorDialog.getColor(current, self, "Pick Color")
        if not color.isValid():
            return
        if which == "gt":
            self.gt_color = color
            self._apply_color_style(self._gt_btn, color)
        else:
            self.sam_color = color
            self._apply_color_style(self._sam_btn, color)
        self._redraw_annotated()

    def _on_opacity_changed(self, val: int):
        self.opacity = val / 100.0
        self._opacity_lbl.setText(f"{val}%")
        self._redraw_annotated()

    # ── dataset / image loading ───────────────────────────────────────────────

    def _pick_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self._load_dataset(path)

    def _switch_user(self):
        if self.dataset is None:
            return
        self._load_dataset(str(self.dataset.root))

    def _load_dataset(self, path: str):
        root = Path(path)
        all_images = Dataset.scan_images(root)
        if not all_images:
            QMessageBox.warning(
                self, "Empty Dataset",
                "No images found.\nMake sure the folder contains an 'images/' sub-directory."
            )
            return

        # First run: no assignment yet → ask who will be auditing
        assignment = Dataset.load_assignment(root)
        if assignment is None:
            dlg = SetupUsersDialog(len(all_images), self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            Dataset.create_assignment(root, all_images, dlg.user_names)

        # Pick which user is working this session
        counts = Dataset.progress_for_all_users(root)
        users = sorted(counts.keys())
        dlg = SelectUserDialog(users, counts, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        user = dlg.selected_user

        try:
            ds = Dataset(path, user)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Cannot load dataset:\n{exc}")
            return

        self.dataset = ds
        self._ds_label.setText(path)
        self._ds_label.setStyleSheet(f"color: {_TEXT};")
        self._user_label.setText(f"  {user}")
        self._switch_btn.setVisible(True)
        self._advance()

    def _advance(self):
        """Display whatever is at the current queue position."""
        if self.dataset is None:
            return
        self._undo_btn.setEnabled(bool(self.dataset._history))
        name = self.dataset.current
        if name is None:
            self._show_done()
            return
        self._load_image(name)

    def _load_image(self, name: str):
        assert self.dataset is not None
        p = self.dataset.get_paths(name)

        def _read(path: Optional[Path], flags=cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
            return cv2.imread(str(path), flags) if path and path.exists() else None

        self._orig_img = _read(p["image"])
        self._gt_mask  = _read(p["mask"],     cv2.IMREAD_GRAYSCALE)
        if p["bbox"] and self._orig_img is not None:
            h, w = self._orig_img.shape[:2]
            self._bboxes = load_bboxes(p["bbox"], w, h)
        else:
            self._bboxes = []
        self._sam_mask = _read(p["sam_mask"], cv2.IMREAD_GRAYSCALE)

        # Reset per-image edit state; keep _edit_mode sticky
        self._sam_mask_edited = None
        self._mask_was_edited = False
        self._reset_btn.setEnabled(False)
        has_sam = self._sam_mask is not None
        self._edit_btn.setEnabled(has_sam)
        if not has_sam and self._edit_mode:
            self._edit_btn.setChecked(False)  # will trigger _toggle_edit_mode(False)

        if self._orig_img is None:
            self._orig_panel.set_image(None)
            self._ann_panel.set_image(None)
        else:
            self._orig_panel.set_image(bgr_to_pixmap(self._orig_img))
            self._redraw_annotated()

        self._img_name_label.setText(name)
        self._refresh_progress()

    def _redraw_annotated(self):
        if self._orig_img is None:
            return
        active_mask = self._sam_mask_edited if self._sam_mask_edited is not None else self._sam_mask
        annotated = compose_overlay(
            self._orig_img,
            self._gt_mask,
            self._bboxes,
            active_mask,
            self._qcolor_to_bgr(self.gt_color),
            self._qcolor_to_bgr(self.sam_color),
            self.opacity,
        )
        self._ann_panel.set_image(bgr_to_pixmap(annotated))

    def _show_done(self):
        self._orig_img = None
        self._orig_panel.set_image(None)
        self._ann_panel.set_image(None)
        self._orig_panel.setText("All images have been audited.")
        self._ann_panel.setText("")
        self._img_name_label.setText("")
        self._refresh_progress()

    def _refresh_progress(self):
        if self.dataset is None:
            self._prog_label.setText("Progress: 0 / 0")
            self._prog_bar.setValue(0)
            self._pct_label.setText("0%")
            return
        done, total = self.dataset.progress
        pct = round(done / total * 100) if total else 0
        self._prog_label.setText(f"Progress: {done} / {total}")
        self._prog_bar.setValue(pct)
        self._pct_label.setText(f"{pct}%")

    # ── edit-mode methods ─────────────────────────────────────────────────────

    def _toggle_edit_mode(self, active: bool):
        self._edit_mode = active
        self._ann_panel.set_edit_mode(active)
        self._edit_btn.setText("✏  Editing…" if active else "✏  Edit Mask")

    def _on_mask_click(self, img_x: int, img_y: int):
        if self._orig_img is None:
            return
        working = self._sam_mask_edited if self._sam_mask_edited is not None else self._sam_mask
        if working is None:
            return
        # Map image-display coords → mask coords (mask may have different resolution)
        mh, mw = working.shape[:2]
        ih, iw = self._orig_img.shape[:2]
        mx = int(img_x / iw * mw)
        my = int(img_y / ih * mh)
        result = erase_component(working, mx, my)
        if result is None:
            return  # clicked on background
        self._sam_mask_edited = result
        self._mask_was_edited = True
        self._reset_btn.setEnabled(True)
        self._redraw_annotated()

    def _reset_edit_state(self, *, keep_mode: bool = False):
        self._sam_mask_edited = None
        self._mask_was_edited = False
        self._reset_btn.setEnabled(False)
        if not keep_mode:
            pass  # mode stays sticky — user controls it via the toggle
        self._redraw_annotated()

    # ── keyboard commands ─────────────────────────────────────────────────────

    def _cmd_segmentation(self):
        if not self.dataset:
            return
        if self._mask_was_edited and self._sam_mask_edited is not None:
            self.dataset.save_cleaned_mask(self.dataset.current, self._sam_mask_edited)
            verdict = "segmentation_cleaned"
        else:
            verdict = "segmentation_original"
        if self.dataset.decide(verdict):
            self._advance()

    def _cmd_nothing(self):
        if self.dataset and self.dataset.decide("nothing"):
            self._advance()

    def _cmd_detection(self):
        if self.dataset and self.dataset.decide("detection"):
            self._advance()

    def _cmd_return_later(self):
        if self.dataset and self.dataset.decide("return_later"):
            self._advance()

    def _cmd_back(self):
        if self.dataset and self.dataset.back():
            self._advance()
            self._undo_btn.setEnabled(bool(self.dataset._history))

    def closeEvent(self, event):
        if self.dataset:
            self.dataset.save()
        event.accept()


# ─────────────────────────────────────────────────────── entry point ──────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MaskLens")
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win._load_dataset(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
