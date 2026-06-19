import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json

from config import Config
from core.task_manager import TaskManager
from schemas.models import TaskStatus

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
os.makedirs(Config.MODELS_FOLDER, exist_ok=True)

task_manager = TaskManager()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "doc-layout-analysis",
        "version": "1.0.0",
    })


@app.route("/analyze", methods=["POST"])
def analyze_document():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "No files selected"}), 400

    ocr_enabled = request.args.get("ocr", "true").lower() == "true"
    output_format = request.args.get("format", "json").lower()

    if output_format not in {"json", "hocr", "alto"}:
        return jsonify({"error": "Invalid output format. Use json, hocr, or alto"}), 400

    try:
        task_ids = task_manager.create_task(
            files=files,
            ocr_enabled=ocr_enabled,
            output_format=output_format,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to create task: {str(e)}"}), 500

    return jsonify({
        "task_ids": task_ids,
        "count": len(task_ids),
        "message": f"{len(task_ids)} task(s) queued for processing",
    }), 202


@app.route("/tasks", methods=["GET"])
def list_tasks():
    tasks = task_manager.get_all_tasks()
    return jsonify({
        "tasks": [t.model_dump() for t in tasks],
        "total": len(tasks),
    })


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    task_info = task_manager.get_task(task_id)
    if not task_info:
        return jsonify({"error": f"Task {task_id} not found"}), 404

    result = task_manager.get_result(task_id)
    output_format = request.args.get("format", "json").lower()

    min_confidence = request.args.get("min_confidence", None)
    if min_confidence is not None:
        try:
            min_confidence = float(min_confidence)
            if min_confidence < 0.0 or min_confidence > 1.0:
                return jsonify({"error": "min_confidence must be between 0 and 1"}), 400
        except ValueError:
            return jsonify({"error": "min_confidence must be a number"}), 400

    response = task_info.model_dump()

    if result and task_info.status == TaskStatus.COMPLETED:
        if output_format == "json":
            result_dict = json.loads(task_manager.pipeline.export_result(result, "json"))

            if min_confidence is not None:
                for page in result_dict.get("pages", []):
                    page["regions"] = [
                        r for r in page.get("regions", [])
                        if r.get("confidence", 0) >= min_confidence
                    ]

            response["result"] = result_dict
        else:
            response["result"] = {
                "format": output_format,
                "content": task_manager.pipeline.export_result(result, output_format),
            }

    return jsonify(response)


@app.route("/tasks/<task_id>/pages/<int:page_number>", methods=["GET"])
def get_page(task_id, page_number):
    task_info = task_manager.get_task(task_id)
    if not task_info:
        return jsonify({"error": f"Task {task_id} not found"}), 404

    page_result = task_manager.get_page_result(task_id, page_number)
    if not page_result:
        return jsonify({"error": f"Page {page_number} not found"}), 404

    return jsonify(page_result)


@app.route("/tasks/<task_id>/export", methods=["GET"])
def export_result(task_id):
    result = task_manager.get_result(task_id)
    if not result:
        return jsonify({"error": f"Result for task {task_id} not found"}), 404

    output_format = request.args.get("format", "json").lower()

    if output_format == "json":
        mimetype = "application/json"
        ext = "json"
    elif output_format == "hocr":
        mimetype = "text/html"
        ext = "hocr"
    elif output_format == "alto":
        mimetype = "application/xml"
        ext = "xml"
    else:
        return jsonify({"error": "Invalid output format"}), 400

    content = task_manager.pipeline.export_result(result, output_format)

    return app.response_class(
        response=content,
        mimetype=mimetype,
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_result.{ext}"
        }
    )


@app.route("/tasks/<task_id>/image/<int:page_number>", methods=["GET"])
def get_page_image(task_id, page_number):
    from utils.file_utils import get_page_image_path

    image_path = get_page_image_path(task_id, page_number)
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image for page {page_number} not found"}), 404

    return send_file(image_path, mimetype="image/png")


@app.route("/tasks/<task_id>", methods=["DELETE"])
def delete_task(task_id):
    success = task_manager.delete_task(task_id)
    if success:
        return jsonify({"message": f"Task {task_id} deleted successfully"}), 200
    else:
        return jsonify({"error": f"Task {task_id} not found"}), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": f"File too large. Maximum size is {Config.MAX_CONTENT_LENGTH // (1024 * 1024)}MB"
    }), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
