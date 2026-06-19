import os
import uuid
import shutil
from typing import List
from datetime import datetime
from config import Config


def generate_task_id() -> str:
    return str(uuid.uuid4())


def get_upload_path(task_id: str, filename: str) -> str:
    task_dir = os.path.join(Config.UPLOAD_FOLDER, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return os.path.join(task_dir, filename)


def get_result_path(task_id: str, filename: str = None) -> str:
    task_dir = os.path.join(Config.RESULTS_FOLDER, task_id)
    os.makedirs(task_dir, exist_ok=True)
    if filename:
        return os.path.join(task_dir, filename)
    return task_dir


def get_page_image_path(task_id: str, page_number: int) -> str:
    return get_result_path(task_id, f"page_{page_number:04d}.png")


def get_analysis_result_path(task_id: str) -> str:
    return get_result_path(task_id, "analysis_result.json")


def save_uploaded_file(file_storage, task_id: str) -> str:
    filename = secure_filename(file_storage.filename)
    file_path = get_upload_path(task_id, filename)
    file_storage.save(file_path)
    return file_path


def secure_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    filename = filename.replace(" ", "_")
    name, ext = os.path.splitext(filename)
    name = "".join(c for c in name if c.isalnum() or c in "-_.")
    if not name:
        name = generate_task_id()[:8]
    return f"{name}{ext.lower()}"


def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower().lstrip(".")


def is_pdf(filename: str) -> bool:
    return get_file_extension(filename) == "pdf"


def is_image(filename: str) -> bool:
    return get_file_extension(filename) in {"jpg", "jpeg", "png", "tiff", "tif"}


def delete_task_data(task_id: str) -> bool:
    upload_dir = os.path.join(Config.UPLOAD_FOLDER, task_id)
    result_dir = os.path.join(Config.RESULTS_FOLDER, task_id)
    success = True
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    return success


def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)


def format_timestamp(dt: datetime = None) -> str:
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def list_files(directory: str, extensions: List[str] = None) -> List[str]:
    if not os.path.exists(directory):
        return []
    files = []
    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        if os.path.isfile(full_path):
            if extensions:
                if get_file_extension(f) in extensions:
                    files.append(full_path)
            else:
                files.append(full_path)
    return sorted(files)
