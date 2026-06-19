import os
import fitz
from PIL import Image
import numpy as np
from typing import List, Tuple
import io
from config import Config
from utils.image_utils import ImageType
from schemas.models import Page


class DocumentLoader:
    @staticmethod
    def load_document(file_path: str, task_id: str) -> List[Page]:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return DocumentLoader._load_pdf(file_path, task_id)
        elif ext in {".jpg", ".jpeg", ".png", ".tiff", ".tif"}:
            return DocumentLoader._load_image(file_path, task_id)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _load_pdf(file_path: str, task_id: str) -> List[Page]:
        pages = []
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=Config.TARGET_DPI)
            img_data = pix.tobytes("png")
            img = np.array(Image.open(io.BytesIO(img_data)))

            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            page_obj = Page(
                page_number=page_num + 1,
                width=pix.width,
                height=pix.height,
                dpi=Config.TARGET_DPI,
            )
            pages.append(page_obj)

            from utils.file_utils import get_page_image_path
            save_path = get_page_image_path(task_id, page_num + 1)
            Image.fromarray(img).save(save_path)
            page_obj.image_path = save_path

        doc.close()
        return pages

    @staticmethod
    def _load_image(file_path: str, task_id: str) -> List[Page]:
        img = np.array(Image.open(file_path))

        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        height, width = img.shape[:2]

        page = Page(
            page_number=1,
            width=width,
            height=height,
            dpi=Config.TARGET_DPI,
        )

        from utils.file_utils import get_page_image_path
        save_path = get_page_image_path(task_id, 1)
        Image.fromarray(img).save(save_path)
        page.image_path = save_path

        return [page]

    @staticmethod
    def get_page_image(page: Page) -> ImageType:
        if not page.image_path or not os.path.exists(page.image_path):
            raise ValueError(f"Image not found for page {page.page_number}")

        from utils.image_utils import load_image
        return load_image(page.image_path)

    @staticmethod
    def get_page_count(file_path: str) -> int:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        else:
            return 1
