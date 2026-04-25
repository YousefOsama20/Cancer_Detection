"""
Cancer Detection GUI Application
=================================
A modern PyQt5-based medical AI application for detecting three types of cancer:
  - Brain Tumor Classification (EfficientNetB0)
  - Breast Cancer Classification (U-Net)
  - Skin Cancer Classification (YOLOv8s-cls)

Author: Cancer Detection Project
"""

# ---------------------------------------------------------------------------
# Dependency Imports
# ---------------------------------------------------------------------------
import sys
import os
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide TF C++ warnings

# IMPORTANT: torch MUST be imported BEFORE PyQt5 on Windows to prevent 
# [WinError 1114] DLL initialization routine failed in c10.dll.
import torch  

import traceback
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QFileDialog, QProgressBar,
    QFrame, QSizePolicy, QGraphicsDropShadowEffect, QMessageBox,
    QSpacerItem, QGroupBox, QGridLayout, QShortcut, QStackedWidget
)
from PyQt5.QtGui import (
    QPixmap, QFont, QColor, QPalette, QIcon, QPainter,
    QLinearGradient, QBrush, QPen, QFontDatabase, QImage
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QTimer, QPropertyAnimation,
    QEasingCurve, pyqtProperty, QRect, QMimeData
)
from PyQt5.QtGui import QKeySequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_TITLE = "MedVision AI — Cancer Detection System"
APP_VERSION = "1.0.0"

# Model paths (relative to this script)
BASE_DIR = Path(__file__).resolve().parent
BRAIN_MODEL_PATH = BASE_DIR / "Brain Tumor_EfficientNetB0" / "Brain Tumor_EfficientNetB0.h5"
BREAST_MODEL_PATH = BASE_DIR / "Breast Cancer U_Net" / "Breast Cancer U_Net.h5"
SKIN_MODEL_PATH = BASE_DIR / "Skin_Cancer_yolo" / "skin_model.pt"

# Class labels (alphabetical order as in training)
BRAIN_TUMOR_CLASSES = ["Glioma", "Healthy", "Meningioma", "Pituitary"]
BREAST_CANCER_CLASSES = ["Benign", "Malignant", "Normal"]
SKIN_CANCER_CLASSES = [
    "Basal Cell Carcinoma",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Pigmented Benign Keratosis",
    "Squamous Cell Carcinoma",
    "Vascular Lesion",
]

# Image sizes
BRAIN_IMG_SIZE = (224, 224)
BREAST_IMG_SIZE = (128, 128)
SKIN_IMG_SIZE = (224, 224)

# Colour palette
COLOR_BG_DARK = "#0d1117"
COLOR_BG_CARD = "#161b22"
COLOR_BG_HOVER = "#1c2333"
COLOR_ACCENT_BLUE = "#58a6ff"
COLOR_ACCENT_GREEN = "#3fb950"
COLOR_ACCENT_RED = "#f85149"
COLOR_ACCENT_ORANGE = "#d29922"
COLOR_ACCENT_PURPLE = "#bc8cff"
COLOR_TEXT_PRIMARY = "#e6edf3"
COLOR_TEXT_SECONDARY = "#8b949e"
COLOR_BORDER = "#30363d"

# Cancer-type specific colours
CANCER_COLORS = {
    "brain": "#58a6ff",    # Blue
    "breast": "#f778ba",   # Pink
    "skin": "#d29922",     # Orange
}

CANCER_ICONS = {
    "brain": "🧠",
    "breast": "🎗️",
    "skin": "🔬",
}

SUPPORTED_FORMATS = "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
def build_stylesheet():
    return f"""
    QMainWindow {{
        background-color: {COLOR_BG_DARK};
    }}
    QWidget {{
        color: {COLOR_TEXT_PRIMARY};
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }}
    QTabWidget::pane {{
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        background-color: {COLOR_BG_DARK};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: {COLOR_BG_CARD};
        color: {COLOR_TEXT_SECONDARY};
        padding: 12px 28px;
        margin-right: 4px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        font-size: 14px;
        font-weight: bold;
        min-width: 160px;
    }}
    QTabBar::tab:selected {{
        background-color: {COLOR_BG_DARK};
        color: {COLOR_TEXT_PRIMARY};
        border: 1px solid {COLOR_BORDER};
        border-bottom: 2px solid {COLOR_ACCENT_BLUE};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {COLOR_BG_HOVER};
        color: {COLOR_TEXT_PRIMARY};
    }}
    QPushButton {{
        background-color: {COLOR_BG_CARD};
        color: {COLOR_TEXT_PRIMARY};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 14px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {COLOR_BG_HOVER};
        border: 1px solid {COLOR_ACCENT_BLUE};
    }}
    QPushButton:pressed {{
        background-color: {COLOR_ACCENT_BLUE};
        color: white;
    }}
    QPushButton:disabled {{
        background-color: {COLOR_BG_CARD};
        color: {COLOR_TEXT_SECONDARY};
        border: 1px solid {COLOR_BORDER};
    }}
    QLabel {{
        color: {COLOR_TEXT_PRIMARY};
    }}
    QProgressBar {{
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        background-color: {COLOR_BG_CARD};
        text-align: center;
        color: {COLOR_TEXT_PRIMARY};
        font-weight: bold;
        height: 24px;
    }}
    QProgressBar::chunk {{
        border-radius: 5px;
    }}
    QGroupBox {{
        border: 1px solid {COLOR_BORDER};
        border-radius: 10px;
        margin-top: 16px;
        padding-top: 20px;
        font-size: 13px;
        font-weight: bold;
        color: {COLOR_TEXT_SECONDARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 16px;
        padding: 0 8px;
    }}
    QMessageBox {{
        background-color: {COLOR_BG_DARK};
    }}
    """


# ---------------------------------------------------------------------------
# Loading spinner widget
# ---------------------------------------------------------------------------
class SpinnerWidget(QWidget):
    """A modern circular loading spinner."""

    def __init__(self, parent=None, color=COLOR_ACCENT_BLUE, size=48):
        super().__init__(parent)
        self._angle = 0
        self._color = QColor(color)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self.setFixedSize(size, size)
        self.hide()

    def _rotate(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    def start(self):
        self.show()
        self._timer.start(30)

    def stop(self):
        self._timer.stop()
        self.hide()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        pen = QPen(QColor(COLOR_BORDER))
        pen.setWidth(4)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        margin = 6
        painter.drawArc(margin, margin, w - 2 * margin, h - 2 * margin, 0, 360 * 16)

        pen.setColor(self._color)
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawArc(margin, margin, w - 2 * margin, h - 2 * margin,
                        self._angle * 16, 90 * 16)
        painter.end()


# ---------------------------------------------------------------------------
# Confidence bar widget
# ---------------------------------------------------------------------------
class ConfidenceBar(QWidget):
    """A single confidence result row with label and animated bar."""

    def __init__(self, label_text, confidence, color, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        lbl = QLabel(label_text)
        lbl.setFixedWidth(220)
        lbl.setFont(QFont("Segoe UI", 11))
        lbl.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        layout.addWidget(lbl)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(confidence * 100))
        bar.setFormat(f"{confidence * 100:.1f}%")
        bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {COLOR_BORDER};
                border-radius: 5px;
                background-color: {COLOR_BG_CARD};
                text-align: center;
                color: {COLOR_TEXT_PRIMARY};
                font-size: 11px;
                font-weight: bold;
                min-height: 22px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(bar)


# ---------------------------------------------------------------------------
# Prediction Worker (runs in QThread)
# ---------------------------------------------------------------------------
class PredictionWorker(QThread):
    """Runs model inference in a background thread."""
    result_ready = pyqtSignal(str, object)   # (predicted_class, payload)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, cancer_type, image_path, parent=None):
        super().__init__(parent)
        self.cancer_type = cancer_type
        self.image_path = image_path

    def run(self):
        try:
            if self.cancer_type == "brain":
                self._predict_brain()
            elif self.cancer_type == "breast":
                self._predict_breast()
            elif self.cancer_type == "skin":
                self._predict_skin()
        except Exception as e:
            self.error_occurred.emit(f"Prediction failed:\n{str(e)}\n\n{traceback.format_exc()}")

    def _predict_brain(self):
        self.status_update.emit("Loading Brain Tumor model…")
        import tensorflow as tf
        from tensorflow.keras.applications.efficientnet import preprocess_input

        model = tf.keras.models.load_model(str(BRAIN_MODEL_PATH), compile=False)
        self.status_update.emit("Preprocessing image…")

        img = tf.keras.utils.load_img(self.image_path, target_size=BRAIN_IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        self.status_update.emit("Running inference…")
        preds = model.predict(img_array, verbose=0)[0]

        results = [(cls, float(conf)) for cls, conf in zip(BRAIN_TUMOR_CLASSES, preds)]
        results.sort(key=lambda x: x[1], reverse=True)
        self.result_ready.emit(results[0][0], results)

    def _predict_breast(self):
        self.status_update.emit("Loading Breast Cancer model…")
        import tensorflow as tf
        import cv2

        model = tf.keras.models.load_model(str(BREAST_MODEL_PATH), compile=False)
        self.status_update.emit("Preprocessing image…")

        img_orig = cv2.imread(self.image_path)
        img_orig = cv2.resize(img_orig, BREAST_IMG_SIZE)
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        img_input = img_gray / 255.0
        img_input = np.expand_dims(img_input, axis=0)  # (1, 128, 128)

        # Determine if model expects a channel dimension
        input_shape = model.input_shape
        if len(input_shape) == 4 and input_shape[-1] == 1:
            img_input = np.expand_dims(img_input, axis=-1)  # (1, 128, 128, 1)

        self.status_update.emit("Running inference…")
        preds = model.predict(img_input, verbose=0)[0]
        mask = preds.squeeze()  # (128, 128)

        # Generate "Union" colored overlay using Jet map
        mask_uint8 = (mask * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        base_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(base_rgb, 0.5, heatmap_rgb, 0.5, 0)

        # Check if a tumor is highlighted 
        tumor_pixels = np.sum(mask > 0.5)
        if tumor_pixels > 0:
            predicted_class = "Tumor Detected (Segmentation Mask)"
        else:
            predicted_class = "No Tumor Detected (Normal)"

        self.result_ready.emit(predicted_class, [("MASK_UNION", overlay)])

    def _predict_skin(self):
        self.status_update.emit("Loading Skin Cancer model (YOLO)…")
        try:
            from ultralytics import YOLO
        except ImportError:
            self.error_occurred.emit(
                "The 'ultralytics' package is required for skin cancer detection.\n"
                "Install it with: pip install ultralytics"
            )
            return

        model = YOLO(str(SKIN_MODEL_PATH))
        self.status_update.emit("Running inference…")
        out = model.predict(self.image_path, imgsz=SKIN_IMG_SIZE[0], verbose=False)

        if out and hasattr(out[0], "probs") and out[0].probs is not None:
            probs = out[0].probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)

            # Get all probabilities
            all_probs = probs.data.cpu().numpy()
            class_names = out[0].names  # dict {0: 'class', ...}

            results = []
            for i, conf in enumerate(all_probs):
                name = class_names.get(i, SKIN_CANCER_CLASSES[i] if i < len(SKIN_CANCER_CLASSES) else f"Class {i}")
                results.append((name, float(conf)))

            results.sort(key=lambda x: x[1], reverse=True)
            self.result_ready.emit(results[0][0], results)
        else:
            self.error_occurred.emit("Could not get predictions from the skin cancer model.")


# ---------------------------------------------------------------------------
# Cancer Detection Tab
# ---------------------------------------------------------------------------
class CancerTab(QWidget):
    """A single tab for one cancer type."""

    def __init__(self, cancer_type, title, description, classes, model_name, color, icon_text, parent=None):
        super().__init__(parent)
        self.cancer_type = cancer_type
        self.classes = classes
        self.color = color
        self.image_path = None
        self._temp_paste_path = None  # Track temp files from clipboard paste
        self.worker = None

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        # --- Header ---
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        icon_label = QLabel(icon_text)
        icon_label.setFont(QFont("Segoe UI Emoji", 36))
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFixedSize(72, 72)
        icon_label.setStyleSheet(f"""
            background-color: {COLOR_BG_CARD};
            border: 2px solid {color};
            border-radius: 16px;
        """)
        header_layout.addWidget(icon_label)

        title_block = QWidget()
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(12, 0, 0, 0)
        title_layout.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title_lbl.setStyleSheet(f"color: {color};")
        title_layout.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setFont(QFont("Segoe UI", 11))
        desc_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        desc_lbl.setWordWrap(True)
        title_layout.addWidget(desc_lbl)

        model_lbl = QLabel(f"📦 Model: {model_name}")
        model_lbl.setFont(QFont("Segoe UI", 10))
        model_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-style: italic;")
        title_layout.addWidget(model_lbl)

        header_layout.addWidget(title_block)
        header_layout.addStretch()
        layout.addWidget(header)

        # --- Body: image + results ---
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setSpacing(20)
        body_layout.setContentsMargins(0, 0, 0, 0)

        # Left: image area
        img_group = QGroupBox("  Medical Image  ")
        img_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                margin-top: 16px;
                padding-top: 24px;
                font-size: 13px;
                font-weight: bold;
                color: {COLOR_TEXT_SECONDARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
            }}
        """)
        img_layout = QVBoxLayout(img_group)
        img_layout.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setFixedSize(360, 360)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"""
            background-color: {COLOR_BG_CARD};
            border: 2px dashed {COLOR_BORDER};
            border-radius: 12px;
            color: {COLOR_TEXT_SECONDARY};
            font-size: 13px;
        """)
        self.image_label.setText("📁\nUpload or paste (Ctrl+V)\nan image to begin")
        img_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # --- Segmented input mode selector ---
        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 8, 0, 0)
        mode_layout.setSpacing(0)

        # Shared style for segmented buttons
        seg_base = f"""
            QPushButton {{
                background-color: {COLOR_BG_CARD};
                color: {COLOR_TEXT_SECONDARY};
                border: 1px solid {COLOR_BORDER};
                font-size: 12px;
                font-weight: bold;
                padding: 8px 18px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_BG_HOVER};
                color: {COLOR_TEXT_PRIMARY};
            }}
        """
        seg_active = f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: 1px solid {color};
                font-size: 12px;
                font-weight: bold;
                padding: 8px 18px;
            }}
        """

        self._seg_base_style = seg_base
        self._seg_active_style = seg_active

        self.seg_upload_btn = QPushButton("📂  Upload")
        self.seg_upload_btn.setMinimumHeight(36)
        self.seg_upload_btn.setStyleSheet(seg_active + "QPushButton { border-top-left-radius: 8px; border-bottom-left-radius: 8px; border-top-right-radius: 0; border-bottom-right-radius: 0; }")
        self.seg_upload_btn.clicked.connect(lambda: self._switch_input_mode("upload"))
        mode_layout.addWidget(self.seg_upload_btn)

        self.seg_paste_btn = QPushButton("📋  Paste")
        self.seg_paste_btn.setMinimumHeight(36)
        self.seg_paste_btn.setStyleSheet(seg_base + "QPushButton { border-top-right-radius: 8px; border-bottom-right-radius: 8px; border-top-left-radius: 0; border-bottom-left-radius: 0; border-left: none; }")
        self.seg_paste_btn.clicked.connect(lambda: self._switch_input_mode("paste"))
        mode_layout.addWidget(self.seg_paste_btn)

        mode_layout.addStretch()
        img_layout.addWidget(mode_row)

        # --- Stacked action area for Upload / Paste ---
        self.input_stack = QStackedWidget()

        # Page 0: Upload action
        upload_page = QWidget()
        upload_page_layout = QHBoxLayout(upload_page)
        upload_page_layout.setContentsMargins(0, 0, 0, 0)
        upload_page_layout.setSpacing(8)

        self.upload_btn = QPushButton("📂  Upload Image")
        self.upload_btn.setMinimumHeight(42)
        self.upload_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
            }}
            QPushButton:hover {{ background-color: {color}cc; }}
            QPushButton:pressed {{ background-color: {color}99; }}
        """)
        self.upload_btn.clicked.connect(self.upload_image)
        upload_page_layout.addWidget(self.upload_btn)
        self.input_stack.addWidget(upload_page)

        # Page 1: Paste action
        paste_page = QWidget()
        paste_page_layout = QVBoxLayout(paste_page)
        paste_page_layout.setContentsMargins(0, 0, 0, 0)
        paste_page_layout.setSpacing(6)

        paste_hint = QLabel("Press  Ctrl+V  or click the button below to paste from clipboard")
        paste_hint.setFont(QFont("Segoe UI", 10))
        paste_hint.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-style: italic;")
        paste_hint.setAlignment(Qt.AlignCenter)
        paste_page_layout.addWidget(paste_hint)

        self.paste_btn = QPushButton("📋  Paste from Clipboard")
        self.paste_btn.setMinimumHeight(42)
        self.paste_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ACCENT_PURPLE};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
            }}
            QPushButton:hover {{ background-color: {COLOR_ACCENT_PURPLE}cc; }}
            QPushButton:pressed {{ background-color: {COLOR_ACCENT_PURPLE}99; }}
        """)
        self.paste_btn.clicked.connect(self.paste_image)
        paste_page_layout.addWidget(self.paste_btn)
        self.input_stack.addWidget(paste_page)

        self.input_stack.setCurrentIndex(0)  # Default to Upload
        img_layout.addWidget(self.input_stack)

        # Buttons row (Clear — always visible)
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 4, 0, 0)

        self.clear_btn = QPushButton("🔄  Clear")
        self.clear_btn.setMinimumHeight(42)
        self.clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.clear_btn)

        # Keyboard shortcut: Ctrl+V to paste
        self._paste_shortcut = QShortcut(QKeySequence("Ctrl+V"), self)
        self._paste_shortcut.activated.connect(self.paste_image)

        img_layout.addWidget(btn_row)
        body_layout.addWidget(img_group, 1)  # stretch factor = 1 (equal 5:5 ratio)

        # Right: results area
        results_group = QGroupBox("  Prediction Results  ")
        results_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                margin-top: 16px;
                padding-top: 24px;
                font-size: 13px;
                font-weight: bold;
                color: {COLOR_TEXT_SECONDARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
            }}
        """)
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(10)

        # Status / spinner area
        spinner_row = QWidget()
        spinner_layout = QHBoxLayout(spinner_row)
        spinner_layout.setContentsMargins(0, 0, 0, 0)
        spinner_layout.setAlignment(Qt.AlignCenter)

        self.spinner = SpinnerWidget(color=color, size=40)
        spinner_layout.addWidget(self.spinner)

        self.status_label = QLabel("Upload an image to begin.")
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        self.status_label.setAlignment(Qt.AlignCenter)
        spinner_layout.addWidget(self.status_label)

        results_layout.addWidget(spinner_row)

        # Predicted class
        self.prediction_label = QLabel("")
        self.prediction_label.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setWordWrap(True)
        self.prediction_label.setStyleSheet(f"color: {color}; margin: 8px 0;")
        self.prediction_label.hide()
        results_layout.addWidget(self.prediction_label)

        # Segmentation mask output (for Breast Cancer)
        self.mask_label = QLabel()
        self.mask_label.setFixedSize(320, 320)
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setStyleSheet(f"""
            background-color: {COLOR_BG_CARD};
            border: 2px dashed {color};
            border-radius: 12px;
        """)
        self.mask_label.hide()
        results_layout.addWidget(self.mask_label, alignment=Qt.AlignCenter)

        # Confidence frame
        self.confidence_frame = QWidget()
        self.confidence_layout = QVBoxLayout(self.confidence_frame)
        self.confidence_layout.setContentsMargins(8, 0, 8, 0)
        self.confidence_layout.setSpacing(6)
        self.confidence_frame.hide()
        results_layout.addWidget(self.confidence_frame)

        # Predict button
        self.predict_btn = QPushButton("⚡  Predict")
        self.predict_btn.setMinimumHeight(48)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ACCENT_GREEN};
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 32px;
            }}
            QPushButton:hover {{ background-color: {COLOR_ACCENT_GREEN}cc; }}
            QPushButton:pressed {{ background-color: {COLOR_ACCENT_GREEN}99; }}
            QPushButton:disabled {{
                background-color: {COLOR_BG_CARD};
                color: {COLOR_TEXT_SECONDARY};
                border: 1px solid {COLOR_BORDER};
            }}
        """)
        self.predict_btn.clicked.connect(self.run_prediction)
        results_layout.addWidget(self.predict_btn)

        results_layout.addStretch()
        body_layout.addWidget(results_group, 1)  # stretch factor = 1 (equal 5:5 ratio)

        layout.addWidget(body)

    # --- Methods ---

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Medical Image", "", SUPPORTED_FORMATS
        )
        if path:
            self.image_path = path
            self._display_image(path)
            self.predict_btn.setEnabled(True)
            self.status_label.setText("Image loaded. Press Predict to analyze.")
            self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_GREEN};")
            self._clear_results()

    def _switch_input_mode(self, mode):
        """Switch between Upload and Paste input modes."""
        if mode == "upload":
            self.input_stack.setCurrentIndex(0)
            self.seg_upload_btn.setStyleSheet(
                self._seg_active_style + "QPushButton { border-top-left-radius: 8px; border-bottom-left-radius: 8px; border-top-right-radius: 0; border-bottom-right-radius: 0; }"
            )
            self.seg_paste_btn.setStyleSheet(
                self._seg_base_style + "QPushButton { border-top-right-radius: 8px; border-bottom-right-radius: 8px; border-top-left-radius: 0; border-bottom-left-radius: 0; border-left: none; }"
            )
        else:
            self.input_stack.setCurrentIndex(1)
            self.seg_paste_btn.setStyleSheet(
                self._seg_active_style + "QPushButton { border-top-right-radius: 8px; border-bottom-right-radius: 8px; border-top-left-radius: 0; border-bottom-left-radius: 0; }"
            )
            self.seg_upload_btn.setStyleSheet(
                self._seg_base_style + "QPushButton { border-top-left-radius: 8px; border-bottom-left-radius: 8px; border-top-right-radius: 0; border-bottom-right-radius: 0; border-right: none; }"
            )

    def paste_image(self):
        """Paste an image from the system clipboard."""
        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()

        pixmap = None
        if mime.hasImage():
            qimage = clipboard.image()
            if not qimage.isNull():
                pixmap = QPixmap.fromImage(qimage)
        elif mime.hasUrls():
            # Some apps put file URLs on clipboard
            for url in mime.urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if os.path.isfile(path):
                        pixmap = QPixmap(path)
                        if not pixmap.isNull():
                            # Use the file directly instead of saving a temp copy
                            self.image_path = path
                            self._display_image(path)
                            self.predict_btn.setEnabled(True)
                            self.status_label.setText("Image pasted from clipboard. Press Predict to analyze.")
                            self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_GREEN};")
                            self._clear_results()
                            return

        if pixmap is None or pixmap.isNull():
            self.status_label.setText("⚠ No image found in clipboard. Copy an image first.")
            self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_ORANGE};")
            return

        # Save the clipboard image to a temp file so the prediction pipeline can use it
        self._cleanup_temp_paste()
        temp_fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="medvision_paste_")
        os.close(temp_fd)
        pixmap.save(temp_path, "PNG")

        self._temp_paste_path = temp_path
        self.image_path = temp_path
        self._display_image(temp_path)
        self.predict_btn.setEnabled(True)
        self.status_label.setText("Image pasted from clipboard. Press Predict to analyze.")
        self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_GREEN};")
        self._clear_results()

    def _cleanup_temp_paste(self):
        """Remove any previously saved temp paste file."""
        if self._temp_paste_path and os.path.isfile(self._temp_paste_path):
            try:
                os.remove(self._temp_paste_path)
            except OSError:
                pass
            self._temp_paste_path = None

    def _display_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.status_label.setText("⚠ Cannot load this image file.")
            self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_RED};")
            self.predict_btn.setEnabled(False)
            return

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setStyleSheet(f"""
            background-color: {COLOR_BG_CARD};
            border: 2px solid {self.color};
            border-radius: 12px;
        """)

    def run_prediction(self):
        if not self.image_path:
            return

        # Validate image exists
        if not os.path.isfile(self.image_path):
            self.status_label.setText("⚠ Image file not found.")
            self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_RED};")
            return

        # Check model file exists
        model_path_map = {
            "brain": BRAIN_MODEL_PATH,
            "breast": BREAST_MODEL_PATH,
            "skin": SKIN_MODEL_PATH,
        }
        mp = model_path_map.get(self.cancer_type)
        if mp and not mp.exists():
            self.status_label.setText(f"⚠ Model file not found:\n{mp.name}")
            self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_RED};")
            QMessageBox.critical(
                self, "Model Not Found",
                f"The model file was not found at:\n{mp}\n\n"
                "Please ensure the model file is in the correct location."
            )
            return

        self.predict_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.paste_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self._clear_results()
        self.spinner.start()
        self.status_label.setText("Initializing…")
        self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_BLUE};")

        self.worker = PredictionWorker(self.cancer_type, self.image_path)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.status_update.connect(self._on_status)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_status(self, msg):
        self.status_label.setText(msg)

    def _on_result(self, predicted_class, results):
        self.spinner.stop()
        self.status_label.setText("✅ Prediction complete!")
        self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_GREEN};")

        self.prediction_label.setText(f"🔎  {predicted_class}")
        self.prediction_label.show()

        # Handle mask image for Breast segmentation
        if self.cancer_type == "breast" and results and results[0][0] in ["MASK_ARRAY", "MASK_UNION"]:
            mask_data = results[0][1]
            if isinstance(mask_data, np.ndarray):
                mask_data = np.ascontiguousarray(mask_data)
                
                if len(mask_data.shape) == 3 and mask_data.shape[2] == 3:
                     # RGB Union
                     h, w, ch = mask_data.shape
                     bytes_per_line = ch * w
                     qimg = QImage(mask_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
                else:
                     # Fallback Grayscale
                     mask_8 = (mask_data * 255.0).astype(np.uint8)
                     mask_8 = np.ascontiguousarray(mask_8)
                     h, w = mask_8.shape
                     qimg = QImage(mask_8.data, w, h, w, QImage.Format_Grayscale8)

                # Ensure the data doesn't get garbage collected until QImage makes a deep copy
                qimg = qimg.copy() 
                pixmap = QPixmap.fromImage(qimg).scaled(
                    320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.mask_label.setPixmap(pixmap)
                self.mask_label.show()
            return

        # Build confidence bars
        top_conf = results[0][1] if results else 0
        for cls_name, conf in results:
            if conf < 0.01 and cls_name != results[0][0]:
                continue  # Skip near-zero
            # Colour picked based on rank
            if cls_name == results[0][0]:
                bar_color = self.color
            elif conf > 0.3:
                bar_color = COLOR_ACCENT_ORANGE
            else:
                bar_color = COLOR_TEXT_SECONDARY

            bar = ConfidenceBar(cls_name, conf, bar_color)
            self.confidence_layout.addWidget(bar)

        self.confidence_frame.show()

    def _on_error(self, msg):
        self.spinner.stop()
        self.status_label.setText("❌ Error occurred.")
        self.status_label.setStyleSheet(f"color: {COLOR_ACCENT_RED};")
        QMessageBox.warning(self, "Prediction Error", msg)

    def _on_finished(self):
        self.predict_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.paste_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

    def _clear_results(self):
        self.prediction_label.hide()
        self.prediction_label.setText("")
        self.mask_label.hide()
        self.mask_label.clear()
        self.confidence_frame.hide()
        # Remove old bars
        while self.confidence_layout.count():
            child = self.confidence_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def clear_all(self):
        self._cleanup_temp_paste()
        self.image_path = None
        self.image_label.clear()
        self.image_label.setText("📁\nUpload or paste (Ctrl+V)\nan image to begin")
        self.image_label.setStyleSheet(f"""
            background-color: {COLOR_BG_CARD};
            border: 2px dashed {COLOR_BORDER};
            border-radius: 12px;
            color: {COLOR_TEXT_SECONDARY};
            font-size: 13px;
        """)
        self.predict_btn.setEnabled(False)
        self.status_label.setText("Upload or paste an image to begin.")
        self.status_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        self._clear_results()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class CancerDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(960, 680)
        self.resize(1100, 750)

        self.setStyleSheet(build_stylesheet())

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(8)

        # --- Top bar ---
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 4, 8, 4)

        logo_lbl = QLabel("🏥")
        logo_lbl.setFont(QFont("Segoe UI Emoji", 24))
        top_layout.addWidget(logo_lbl)

        app_title = QLabel(APP_TITLE)
        app_title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        app_title.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        top_layout.addWidget(app_title)

        top_layout.addStretch()

        version_lbl = QLabel(f"v{APP_VERSION}")
        version_lbl.setFont(QFont("Segoe UI", 10))
        version_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        top_layout.addWidget(version_lbl)

        main_layout.addWidget(top_bar)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {COLOR_BORDER};")
        main_layout.addWidget(sep)

        # --- Tab widget ---
        self.tabs = QTabWidget()

        # Brain Tumor tab
        brain_tab = CancerTab(
            cancer_type="brain",
            title="Brain Tumor Detection",
            description="Classifies brain MRI images into 4 categories: Glioma, Meningioma, Pituitary tumor, or Healthy.",
            classes=BRAIN_TUMOR_CLASSES,
            model_name="EfficientNetB0 (Keras .h5)",
            color=CANCER_COLORS["brain"],
            icon_text=CANCER_ICONS["brain"],
        )
        self.tabs.addTab(brain_tab, f"  {CANCER_ICONS['brain']}  Brain Tumor  ")

        # Breast Cancer tab
        breast_tab = CancerTab(
            cancer_type="breast",
            title="Breast Cancer Detection",
            description="Performs UNet image segmentation to highlight and locate tumors in breast ultrasound images.",
            classes=BREAST_CANCER_CLASSES,
            model_name="U-Net Segmentation (Keras .h5)",
            color=CANCER_COLORS["breast"],
            icon_text=CANCER_ICONS["breast"],
        )
        self.tabs.addTab(breast_tab, f"  {CANCER_ICONS['breast']}  Breast Cancer  ")

        # Skin Cancer tab
        skin_tab = CancerTab(
            cancer_type="skin",
            title="Skin Cancer Detection",
            description="Classifies dermoscopic skin images into 7 categories using a YOLOv8 classifier.",
            classes=SKIN_CANCER_CLASSES,
            model_name="YOLOv8s-cls (Ultralytics)",
            color=CANCER_COLORS["skin"],
            icon_text=CANCER_ICONS["skin"],
        )
        self.tabs.addTab(skin_tab, f"  {CANCER_ICONS['skin']}  Skin Cancer  ")

        main_layout.addWidget(self.tabs)

        # --- Status bar ---
        self.statusBar().setStyleSheet(
            f"background-color: {COLOR_BG_CARD}; color: {COLOR_TEXT_SECONDARY}; "
            f"border-top: 1px solid {COLOR_BORDER}; padding: 4px 12px; font-size: 11px;"
        )
        self.statusBar().showMessage(
            "Ready  •  Select a cancer type tab and upload a medical image to begin analysis."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette base
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLOR_BG_DARK))
    palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(COLOR_BG_CARD))
    palette.setColor(QPalette.AlternateBase, QColor(COLOR_BG_HOVER))
    palette.setColor(QPalette.ToolTipBase, QColor(COLOR_BG_CARD))
    palette.setColor(QPalette.ToolTipText, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.Text, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(COLOR_BG_CARD))
    palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.Highlight, QColor(COLOR_ACCENT_BLUE))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = CancerDetectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
