import cv2
import numpy as np
from typing import Tuple, Optional
import numpy.typing as npt


ImageType = npt.NDArray[np.uint8]


def load_image(file_path: str) -> ImageType:
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {file_path}")
    return img


def save_image(file_path: str, img: ImageType) -> None:
    cv2.imwrite(file_path, img)


def normalize_image(img: ImageType, target_dpi: int, original_dpi: int = 72) -> ImageType:
    if original_dpi <= 0 or target_dpi <= 0 or original_dpi == target_dpi:
        return img
    scale = target_dpi / original_dpi
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def denoise_image(img: ImageType, kernel_size: int = 3) -> ImageType:
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def binarize_image(img: ImageType) -> ImageType:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def deskew_image(img: ImageType, max_angle: float = 45.0) -> Tuple[ImageType, float]:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return img, 0.0

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > max_angle:
        angle = max_angle if angle > 0 else -max_angle

    if abs(angle) < 0.5:
        return img, 0.0

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated, angle


def detect_edges(img: ImageType, low_threshold: int = 50, high_threshold: int = 150) -> ImageType:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.Canny(gray, low_threshold, high_threshold)


def detect_lines(edges: ImageType, threshold: int = 100,
                 min_line_length: int = 100, max_line_gap: int = 10) -> list:
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return []
    return lines.reshape(-1, 4).tolist()


def draw_regions(img: ImageType, regions: list, colors: dict, alpha: float = 0.4) -> ImageType:
    overlay = img.copy()
    for region in regions:
        color = colors.get(region.type.value, (128, 128, 128))
        bbox = region.bbox.to_absolute(img.shape[1], img.shape[0])
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        if region.reading_order is not None:
            label = f"{region.type.value} #{region.reading_order}"
        else:
            label = region.type.value

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(overlay, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), thickness)

    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    for region in regions:
        color = colors.get(region.type.value, (128, 128, 128))
        bbox = region.bbox.to_absolute(img.shape[1], img.shape[0])
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

    return result


def crop_region(img: ImageType, bbox: tuple, padding: int = 0) -> ImageType:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.shape[1], x2 + padding)
    y2 = min(img.shape[0], y2 + padding)
    return img[y1:y2, x1:x2]


def compute_iou(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def preprocess_for_inference(img: ImageType, target_size: Tuple[int, int] = (1024, 1024)) -> ImageType:
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
    elif resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    return normalized.transpose(2, 0, 1)
