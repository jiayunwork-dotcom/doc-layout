#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("文档布局分析系统 - 模块导入测试")
print("=" * 60)

modules = [
    ("config", "配置模块"),
    ("schemas.models", "数据模型"),
    ("utils.image_utils", "图像处理工具"),
    ("utils.file_utils", "文件工具"),
    ("processing.document_loader", "文档加载器"),
    ("processing.preprocessor", "文档预处理器"),
    ("processing.ocr_engine", "OCR引擎"),
    ("processing.xy_cut", "XY-Cut算法"),
    ("processing.reading_order", "阅读顺序推断"),
    ("processing.hierarchy", "层级结构提取"),
    ("models.layout_detector", "布局检测器"),
    ("models.table_recognizer", "表格识别器"),
    ("output.json_exporter", "JSON导出"),
    ("output.hocr_exporter", "hOCR导出"),
    ("output.alto_exporter", "ALTO导出"),
    ("core.metrics", "评估指标"),
    ("core.pipeline", "分析流水线"),
    ("core.task_manager", "任务管理器"),
]

failed = []

for module_name, description in modules:
    try:
        __import__(module_name)
        print(f"✅ {description:20s} - {module_name}")
    except Exception as e:
        print(f"❌ {description:20s} - {module_name}: {e}")
        failed.append((module_name, str(e)))

print("=" * 60)
if failed:
    print(f"有 {len(failed)} 个模块导入失败")
    for name, err in failed:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("所有模块导入成功！")
    print("\n系统架构验证完成。")
