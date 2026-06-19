import os
import cv2
import numpy as np
from typing import Optional
try:
    import pytesseract
except ImportError:
    pytesseract = None

from utils.image_utils import ImageType, crop_region
from schemas.models import Region, BBox
from config import Config


class OCREngine:
    def __init__(self, enabled: bool = True, lang: str = "chi_sim+eng"):
        self.enabled = enabled and pytesseract is not None
        self.lang = lang

    def extract_text(self, img: ImageType, region: Optional[Region] = None) -> str:
        if not self.enabled:
            return ""

        try:
            if region is not None:
                bbox = region.bbox.to_absolute(img.shape[1], img.shape[0])
                img_crop = crop_region(img, bbox, padding=5)
            else:
                img_crop = img

            if len(img_crop.shape) == 3:
                gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_crop

            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            kernel = np.ones((1, 1), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            text = pytesseract.image_to_string(gray, lang=self.lang, config="--psm 6")
            return text.strip()

        except Exception as e:
            return ""

    def extract_text_regions(self, img: ImageType) -> list:
        if not self.enabled:
            return []

        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            data = pytesseract.image_to_data(
                gray, lang=self.lang, output_type=pytesseract.Output.DICT,
                config="--psm 6"
            )

            regions = []
            h, w = img.shape[:2]

            n_boxes = len(data["text"])
            for i in range(n_boxes):
                if int(data["conf"][i]) > 60 and data["text"][i].strip():
                    x = data["left"][i] / w
                    y = data["top"][i] / h
                    width = data["width"][i] / w
                    height = data["height"][i] / h

                    regions.append({
                        "bbox": BBox(x=x, y=y, width=width, height=height),
                        "text": data["text"][i],
                        "confidence": float(data["conf"][i]) / 100.0,
                    })

            return regions

        except Exception as e:
            return []

    def is_available(self) -> bool:
        if not self.enabled:
            return False
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
