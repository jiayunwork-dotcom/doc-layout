import cv2
import numpy as np
from typing import List, Tuple
from utils.image_utils import (
    ImageType,
    deskew_image,
    denoise_image,
    binarize_image,
    normalize_image,
    save_image,
)
from schemas.models import Page
from config import Config


class DocumentPreprocessor:
    def __init__(self, enable_deskew: bool = True, enable_denoise: bool = False,
                 enable_binarize: bool = False):
        self.enable_deskew = enable_deskew
        self.enable_denoise = enable_denoise
        self.enable_binarize = enable_binarize

    def preprocess_pages(self, pages: List[Page], task_id: str) -> List[Page]:
        for page in pages:
            self._preprocess_page(page, task_id)
        return pages

    def _preprocess_page(self, page: Page, task_id: str) -> None:
        if not page.image_path:
            return

        img = cv2.imread(page.image_path, cv2.IMREAD_COLOR)
        if img is None:
            return

        original_dpi = page.dpi
        applied = []

        if self.enable_deskew:
            img, angle = deskew_image(img)
            if abs(angle) > 0.1:
                applied.append(f"deskew ({angle:.1f}°)")

        if self.enable_denoise:
            img = denoise_image(img)
            applied.append("denoise")

        if self.enable_binarize:
            img = binarize_image(img)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            applied.append("binarize")

        img = normalize_image(img, Config.TARGET_DPI, original_dpi)
        if original_dpi != Config.TARGET_DPI:
            applied.append(f"normalize_dpi ({Config.TARGET_DPI})")
            page.dpi = Config.TARGET_DPI
            page.height, page.width = img.shape[:2]

        save_image(page.image_path, img)
        page.preprocessing_applied = applied

    @staticmethod
    def detect_scan_quality(img: ImageType) -> dict:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        contrast = gray.std()

        brightness = gray.mean() / 255.0

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_density = (binary > 0).sum() / binary.size

        return {
            "blur_score": float(blur_score),
            "contrast": float(contrast),
            "brightness": float(brightness),
            "text_density": float(text_density),
            "needs_denoising": blur_score < 100,
            "needs_enhancement": contrast < 50 or brightness < 0.2 or brightness > 0.8,
        }

    @staticmethod
    def estimate_skew_angle(img: ImageType) -> float:
        _, angle = deskew_image(img)
        return angle
