import os
from typing import List


class Config:
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "data/uploads")
    RESULTS_FOLDER = os.environ.get("RESULTS_FOLDER", "data/results")
    MODELS_FOLDER = os.environ.get("MODELS_FOLDER", "data/models")
    MAX_BATCH_FILES = int(os.environ.get("MAX_BATCH_FILES", 10))
    BATCH_INFERENCE_SIZE = int(os.environ.get("BATCH_SIZE", 4))
    MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_FOLDER, "layout_model.onnx"))
    OCR_ENABLED = os.environ.get("OCR_ENABLED", "true").lower() == "true"
    TARGET_DPI = int(os.environ.get("TARGET_DPI", 300))
    ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png", "tiff", "tif"}
    REGION_COLORS = {
        "text": (135, 206, 250),
        "title_h1": (255, 99, 71),
        "title_h2": (255, 165, 0),
        "title_h3": (255, 215, 0),
        "table": (144, 238, 144),
        "figure": (218, 112, 214),
        "caption": (75, 0, 130),
        "header": (70, 130, 180),
        "footer": (70, 130, 180),
        "sidebar": (244, 164, 96),
        "formula": (152, 251, 152),
        "formula_inline": (152, 251, 152),
        "list": (176, 224, 230),
    }

    @staticmethod
    def allowed_file(filename: str) -> bool:
        return "." in filename and \
               filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    @staticmethod
    def get_region_color(region_type: str) -> tuple:
        return Config.REGION_COLORS.get(region_type, (128, 128, 128))
