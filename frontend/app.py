import streamlit as st
import requests
import os
import time
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import base64

API_URL = os.environ.get("API_URL", "http://localhost:5000")

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

REGION_NAMES = {
    "text": "正文",
    "title_h1": "一级标题",
    "title_h2": "二级标题",
    "title_h3": "三级标题",
    "table": "表格",
    "figure": "图片",
    "caption": "图注/表注",
    "header": "页眉",
    "footer": "页脚",
    "sidebar": "侧栏",
    "formula": "数学公式",
    "formula_inline": "行内公式",
    "list": "列表",
}

st.set_page_config(
    page_title="文档布局分析演示",
    page_icon="📄",
    layout="wide",
)

st.title("📄 文档布局分析与版面区域分割")
st.markdown("---")


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_files(files, ocr_enabled=True, output_format="json"):
    upload_files = [("files", (f.name, f, f.type)) for f in files]
    params = {"ocr": str(ocr_enabled).lower(), "format": output_format}
    try:
        response = requests.post(f"{API_URL}/analyze", files=upload_files, params=params, timeout=30)
        return response.json() if response.status_code == 202 else None
    except Exception as e:
        st.error(f"上传失败: {e}")
        return None


def get_task_status(task_id):
    try:
        response = requests.get(f"{API_URL}/tasks/{task_id}", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_page_result(task_id, page_number):
    try:
        response = requests.get(f"{API_URL}/tasks/{task_id}/pages/{page_number}", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_page_image(task_id, page_number):
    try:
        response = requests.get(f"{API_URL}/tasks/{task_id}/image/{page_number}", timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except:
        return None


def list_tasks():
    try:
        response = requests.get(f"{API_URL}/tasks", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def delete_task(task_id):
    try:
        response = requests.delete(f"{API_URL}/tasks/{task_id}", timeout=10)
        return response.status_code == 200
    except:
        return False


def create_comparison(source_id, target_id, page_number=None):
    payload = {"source_id": source_id, "target_id": target_id}
    if page_number is not None:
        payload["page_number"] = page_number
    try:
        response = requests.post(f"{API_URL}/compare", json=payload, timeout=30)
        return response.json() if response.status_code == 202 else None
    except Exception as e:
        st.error(f"创建比对失败: {e}")
        return None


def get_comparison_status(comparison_id):
    try:
        response = requests.get(f"{API_URL}/comparisons/{comparison_id}", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def list_comparisons():
    try:
        response = requests.get(f"{API_URL}/comparisons", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def export_comparison(comparison_id):
    try:
        response = requests.get(f"{API_URL}/comparisons/{comparison_id}/export", timeout=30)
        return response if response.status_code == 200 else None
    except:
        return None


DIFF_TYPE_COLORS = {
    "added": (34, 197, 94),
    "removed": (239, 68, 68),
    "moved": (234, 179, 8),
    "modified": (168, 85, 247),
    "unchanged": (107, 114, 128),
}

DIFF_TYPE_NAMES = {
    "added": "新增",
    "removed": "删除",
    "moved": "移动",
    "modified": "修改",
    "unchanged": "未变化",
}


def draw_diff_regions_on_image(image, diffs, side="source", show_unchanged=True):
    from PIL import ImageDraw, ImageFont

    img = image.copy().convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 14)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

    for diff in diffs:
        diff_type = diff["type"]

        if diff_type == "unchanged" and not show_unchanged:
            continue

        if side == "source":
            region = diff.get("source_region")
        else:
            region = diff.get("target_region")

        if not region:
            continue

        bbox = region["bbox"]
        color = DIFF_TYPE_COLORS.get(diff_type, (128, 128, 128))

        x1 = int(bbox["x"] * w)
        y1 = int(bbox["y"] * h)
        x2 = int((bbox["x"] + bbox["width"]) * w)
        y2 = int((bbox["y"] + bbox["height"]) * h)

        if diff_type == "unchanged":
            alpha = 40
        else:
            alpha = 100

        fill_color = (color[0], color[1], color[2], alpha)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        line_width = 2 if diff_type == "unchanged" else 3
        outline_color = (color[0], color[1], color[2], 255)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=line_width)

        label = DIFF_TYPE_NAMES.get(diff_type, diff_type)
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except:
            text_w, text_h = len(label) * 8, 16

        label_y = y1 - text_h - 10
        if label_y < 0:
            label_y = y1 + 5

        draw.rectangle(
            [x1, label_y, x1 + text_w + 10, label_y + text_h + 8],
            fill=outline_color
        )
        draw.text((x1 + 5, label_y + 3), label, fill=(255, 255, 255, 255), font=font)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def draw_moved_connections(source_image, target_image, diffs):
    from PIL import ImageDraw

    src_img = source_image.copy().convert("RGBA")
    tgt_img = target_image.copy().convert("RGBA")
    src_w, src_h = src_img.size
    tgt_w, tgt_h = tgt_img.size

    total_width = src_w + tgt_w
    total_height = max(src_h, tgt_h)

    combined = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
    combined.paste(src_img, (0, 0))
    combined.paste(tgt_img, (src_w, 0))

    draw = ImageDraw.Draw(combined)

    for diff in diffs:
        if diff["type"] != "moved":
            continue

        src_region = diff.get("source_region")
        tgt_region = diff.get("target_region")

        if not src_region or not tgt_region:
            continue

        src_bbox = src_region["bbox"]
        tgt_bbox = tgt_region["bbox"]

        src_center_x = int((src_bbox["x"] + src_bbox["width"] / 2) * src_w)
        src_center_y = int((src_bbox["y"] + src_bbox["height"] / 2) * src_h)

        tgt_center_x = int((tgt_bbox["x"] + tgt_bbox["width"] / 2) * tgt_w) + src_w
        tgt_center_y = int((tgt_bbox["y"] + tgt_bbox["height"] / 2) * tgt_h)

        color = DIFF_TYPE_COLORS["moved"]
        draw.line(
            [(src_center_x, src_center_y), (tgt_center_x, tgt_center_y)],
            fill=(color[0], color[1], color[2], 200),
            width=2,
            joint="curve"
        )

        for i in range(0, int(((tgt_center_x - src_center_x) ** 2 + (tgt_center_y - src_center_y) ** 2) ** 0.5), 15):
            t = i / int(((tgt_center_x - src_center_x) ** 2 + (tgt_center_y - src_center_y) ** 2) ** 0.5)
            dash_x = src_center_x + t * (tgt_center_x - src_center_x)
            dash_y = src_center_y + t * (tgt_center_y - src_center_y)
            draw.ellipse(
                [dash_x - 2, dash_y - 2, dash_x + 2, dash_y + 2],
                fill=(color[0], color[1], color[2], 255)
            )

    return combined.convert("RGB")


def draw_regions_on_image(image, regions, selected_region_id=None, page_width=None, page_height=None):
    from PIL import ImageDraw, ImageFont
    import math

    img = image.copy().convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 14)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

    try:
        font_bold = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 14)
    except:
        font_bold = font

    for region in regions:
        bbox = region["bbox"]
        rtype = region["type"]
        color = REGION_COLORS.get(rtype, (128, 128, 128))

        x1 = int(bbox["x"] * w)
        y1 = int(bbox["y"] * h)
        x2 = int((bbox["x"] + bbox["width"]) * w)
        y2 = int((bbox["y"] + bbox["height"]) * h)

        is_selected = selected_region_id and region["id"] == selected_region_id
        alpha = 100 if not is_selected else 160

        fill_color = (color[0], color[1], color[2], alpha)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        line_width = 4 if is_selected else 2
        outline_color = (color[0], color[1], color[2], 255)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=line_width)

        label = REGION_NAMES.get(rtype, rtype)
        if region.get("reading_order"):
            label = f"{label} #{region['reading_order']}"

        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except:
            text_w, text_h = len(label) * 8, 16

        label_y = y1 - text_h - 10
        if label_y < 0:
            label_y = y1 + 5

        draw.rectangle(
            [x1, label_y, x1 + text_w + 10, label_y + text_h + 8],
            fill=outline_color
        )

        draw.text((x1 + 5, label_y + 3), label, fill=(255, 255, 255, 255), font=font)

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def display_region_details(region, page_image=None):
    st.markdown(f"### 区域详情")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**类型**: {REGION_NAMES.get(region['type'], region['type'])}")
        st.write(f"**置信度**: {region['confidence']:.4f}")
        if region.get("reading_order"):
            st.write(f"**阅读顺序**: #{region['reading_order']}")

    with col2:
        bbox = region["bbox"]
        st.write(f"**位置**: x={bbox['x']:.4f}, y={bbox['y']:.4f}")
        st.write(f"**大小**: w={bbox['width']:.4f}, h={bbox['height']:.4f}")

    if region.get("text"):
        st.markdown("#### 文本内容")
        st.text_area("", region["text"], height=150, disabled=True)

    if region.get("table_structure"):
        st.markdown("#### 表格结构")
        ts = region["table_structure"]
        st.write(f"**行列数**: {ts['rows']} 行 × {ts['cols']} 列")

        if ts.get("grid"):
            df = pd.DataFrame(ts["grid"])
            df.columns = [f"列{i+1}" for i in range(df.shape[1])]
            df.index = [f"行{i+1}" for i in range(df.shape[0])]
            st.dataframe(df, use_container_width=True, height=400)

        if ts.get("cells"):
            with st.expander("查看单元格详情"):
                for cell in ts["cells"]:
                    st.write(
                        f"单元格 (行{cell['row_index']+1}, 列{cell['col_index']+1}) "
                        f"- 跨行: {cell['row_span']}, 跨列: {cell['col_span']}"
                    )
                    if cell.get("text"):
                        st.text(cell["text"])
                    st.divider()

    if page_image:
        st.markdown("#### 区域裁剪")
        h, w = np.array(page_image).shape[:2]
        bbox = region["bbox"]
        x1 = int(bbox["x"] * w)
        y1 = int(bbox["y"] * h)
        x2 = int((bbox["x"] + bbox["width"]) * w)
        y2 = int((bbox["y"] + bbox["height"]) * h)
        cropped = page_image.crop((x1, y1, x2, y2))
        st.image(cropped, use_container_width=True)


api_healthy = check_api_health()

with st.sidebar:
    st.header("⚙️ 设置")

    if api_healthy:
        st.success("✅ API服务连接正常")
    else:
        st.error("❌ API服务连接失败")
        st.info(f"当前API地址: {API_URL}")

    st.divider()

    ocr_enabled = st.toggle("启用OCR", value=True, help="对识别出的文本区域进行文字识别")
    output_format = st.selectbox("输出格式", ["json", "hocr", "alto"], index=0)

    st.divider()
    st.header("🎯 置信度过滤")
    min_confidence = st.slider(
        "最低置信度",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="只显示置信度大于等于此阈值的区域"
    )
    if "confidence_filter_info" in st.session_state:
        st.info(st.session_state["confidence_filter_info"])

    st.divider()
    st.header("📋 任务列表")

    tasks_data = list_tasks()
    if tasks_data and tasks_data.get("tasks"):
        tasks = tasks_data["tasks"]
        st.info(f"共 {tasks_data['total']} 个任务")

        for task in reversed(tasks[-10:]):
            status_emoji = {
                "pending": "⏳",
                "processing": "⚙️",
                "completed": "✅",
                "failed": "❌",
            }.get(task["status"], "❓")

            with st.expander(f"{status_emoji} {task['task_id'][:8]}..."):
                st.write(f"**状态**: {task['status']}")
                if task.get("metadata"):
                    st.write(f"**文件**: {task['metadata']['filename']}")
                st.write(f"**进度**: {task['progress']*100:.0f}%")
                if task.get("message"):
                    st.caption(task["message"])
                st.write(f"**创建时间**: {task['created_at']}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("查看", key=f"view_{task['task_id']}"):
                        st.session_state["current_task_id"] = task["task_id"]
                with col2:
                    if st.button("删除", key=f"del_{task['task_id']}"):
                        if delete_task(task["task_id"]):
                            st.success("已删除")
                            st.rerun()
                        else:
                            st.error("删除失败")
    else:
        st.info("暂无历史任务")


tab1, tab2, tab3, tab4 = st.tabs(["📤 上传文档", "🔍 分析结果", "📊 版面比对", "ℹ️ 使用说明"])

with tab1:
    st.header("上传文档进行分析")
    st.markdown("支持PDF和图片格式（JPG/PNG），支持批量上传（最多10个文件），单文件不超过50MB")

    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "jpg", "jpeg", "png", "tiff", "tif"],
        accept_multiple_files=True,
        help="可同时选择多个文件"
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button("🚀 开始分析", type="primary", disabled=not uploaded_files or not api_healthy)
    with col2:
        if st.button("🗑️ 清除选择", disabled=not uploaded_files):
            st.rerun()

    if analyze_btn and uploaded_files:
        with st.status("正在上传并创建任务...", expanded=True) as status:
            result = upload_files(uploaded_files, ocr_enabled, output_format)
            if result and result.get("task_ids"):
                status.update(label=f"✅ 已创建 {result['count']} 个分析任务", state="complete")
                st.success(result["message"])
                st.write(f"任务ID: {', '.join(result['task_ids'])}")
                if len(result["task_ids"]) == 1:
                    st.session_state["current_task_id"] = result["task_ids"][0]
                st.balloons()
            else:
                status.update(label="❌ 创建任务失败", state="error")

with tab2:
    st.header("分析结果")

    task_id = st.text_input("输入任务ID", value=st.session_state.get("current_task_id", ""),
                            placeholder="输入或从侧边栏选择任务ID")

    if not task_id:
        st.info("请输入任务ID或从侧边栏选择任务")
    else:
        task_data = get_task_status(task_id)

        if not task_data:
            st.error("任务不存在")
        else:
            status = task_data["status"]
            progress = task_data["progress"]

            st.subheader(f"任务状态: {status}")
            progress_bar = st.progress(progress)
            if task_data.get("message"):
                st.caption(task_data["message"])

            if status == "pending":
                st.info("任务排队中，请等待...")
                time.sleep(2)
                st.rerun()

            elif status == "processing":
                st.info("分析进行中，正在刷新状态...")
                time.sleep(2)
                st.rerun()

            elif status == "completed" and task_data.get("result"):
                result = task_data["result"]
                pages = result.get("pages", [])

                if not pages:
                    st.warning("没有分析结果")
                else:
                    total_pages = len(pages)
                    page_num = st.slider("选择页码", 1, total_pages, 1)

                    col1, col2 = st.columns([3, 2])

                    with col1:
                        page_data = get_page_result(task_id, page_num)
                        page_image = get_page_image(task_id, page_num)

                        if page_data and page_image:
                            regions = page_data.get("regions", [])
                            sorted_regions = sorted(
                                regions,
                                key=lambda r: r.get("reading_order", 999)
                            )

                            total_region_count = len(sorted_regions)
                            confidence_filtered_regions = [
                                r for r in sorted_regions
                                if r.get("confidence", 0) >= min_confidence
                            ]
                            confidence_filtered_count = len(confidence_filtered_regions)

                            st.session_state["confidence_filter_info"] = f"显示 {confidence_filtered_count}/{total_region_count} 个区域"

                            selected_region_id = st.session_state.get("selected_region_id")
                            annotated_image = draw_regions_on_image(
                                page_image, confidence_filtered_regions, selected_region_id
                            )

                            st.image(annotated_image, caption=f"第 {page_num} 页 / 共 {total_pages} 页",
                                     use_container_width=True)

                            if page_data.get("preprocessing_applied"):
                                st.caption(f"预处理: {', '.join(page_data['preprocessing_applied'])}")

                    with col2:
                        st.markdown("### 📊 区域列表")

                        type_filter = st.multiselect(
                            "筛选类型",
                            list(REGION_NAMES.keys()),
                            default=list(REGION_NAMES.keys()),
                            format_func=lambda x: REGION_NAMES[x]
                        )

                        filtered_regions = [
                            r for r in confidence_filtered_regions if r["type"] in type_filter
                        ]

                        st.info(f"当前列表 {len(filtered_regions)} 个区域")

                        for idx, region in enumerate(filtered_regions):
                            rtype = region["type"]
                            color = REGION_COLORS.get(rtype, (128, 128, 128))
                            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                            order_text = f"#{region.get('reading_order', '?')}" if region.get("reading_order") else ""
                            label = f"{order_text} {REGION_NAMES.get(rtype, rtype)} - {region['confidence']:.2f}"

                            if st.button(
                                label,
                                key=f"region_{region['id']}",
                                use_container_width=True,
                                type="primary" if st.session_state.get("selected_region_id") == region["id"] else "secondary"
                            ):
                                st.session_state["selected_region_id"] = region["id"]
                                st.rerun()

                        if st.session_state.get("selected_region_id"):
                            selected_region = next(
                                (r for r in sorted_regions if r["id"] == st.session_state["selected_region_id"]),
                                None
                            )
                            if selected_region:
                                st.divider()
                                display_region_details(selected_region, page_image)

                    st.divider()

                    st.markdown("### 📁 导出结果")
                    export_col1, export_col2, export_col3 = st.columns(3)

                    with export_col1:
                        if st.button("📄 导出 JSON", use_container_width=True):
                            st.json(result)

                    with export_col2:
                        if st.button("📝 导出 hOCR", use_container_width=True):
                            export_response = requests.get(f"{API_URL}/tasks/{task_id}/export", params={"format": "hocr"})
                            if export_response.status_code == 200:
                                st.download_button(
                                    "⬇️ 下载 hOCR",
                                    export_response.content,
                                    file_name=f"{task_id}_result.hocr",
                                    mime="text/html"
                                )

                    with export_col3:
                        if st.button("📋 导出 ALTO XML", use_container_width=True):
                            export_response = requests.get(f"{API_URL}/tasks/{task_id}/export", params={"format": "alto"})
                            if export_response.status_code == 200:
                                st.download_button(
                                    "⬇️ 下载 ALTO XML",
                                    export_response.content,
                                    file_name=f"{task_id}_result.xml",
                                    mime="application/xml"
                                )

                    if result.get("evaluation"):
                        st.divider()
                        st.markdown("### 📈 评估指标")
                        eval_data = result["evaluation"]
                        eval_col1, eval_col2 = st.columns(2)
                        with eval_col1:
                            st.metric("mAP", f"{eval_data.get('mAP', 0):.4f}" if eval_data.get('mAP') else "N/A")
                        with eval_col2:
                            st.metric("Mean IoU", f"{eval_data.get('mean_iou', 0):.4f}" if eval_data.get('mean_iou') else "N/A")

                        if eval_data.get("per_class_iou"):
                            with st.expander("查看各类别指标"):
                                iou_df = pd.DataFrame.from_dict(
                                    eval_data["per_class_iou"],
                                    orient="index",
                                    columns=["IoU"]
                                )
                                iou_df.index = [REGION_NAMES.get(idx, idx) for idx in iou_df.index]
                                st.bar_chart(iou_df)

            elif status == "failed":
                st.error(f"分析失败: {task_data.get('error', '未知错误')}")

with tab3:
    st.header("📊 版面比对")
    st.markdown("对同一份文档的两个版本进行区域级差异分析，标记新增、删除、移动和修改的区域")

    st.divider()

    completed_tasks = []
    tasks_data = list_tasks()
    if tasks_data and tasks_data.get("tasks"):
        completed_tasks = [
            t for t in tasks_data["tasks"]
            if t["status"] == "completed"
        ]

    if not completed_tasks:
        st.info("暂无已完成的分析任务，请先上传并分析文档")
    else:
        task_options = {
            f"{t['metadata']['filename']} ({t['task_id'][:8]}...)": t["task_id"]
            for t in completed_tasks
        }

        col1, col2 = st.columns(2)
        with col1:
            source_task_label = st.selectbox(
                "选择源版本 (Source)",
                list(task_options.keys()),
                key="source_task_select",
                format_func=lambda x: x
            )
            source_id = task_options[source_task_label]

        with col2:
            target_task_label = st.selectbox(
                "选择目标版本 (Target)",
                list(task_options.keys()),
                key="target_task_select",
                format_func=lambda x: x,
                index=min(1, len(task_options) - 1) if len(task_options) > 1 else 0
            )
            target_id = task_options[target_task_label]

        page_col1, page_col2, page_col3 = st.columns([1, 1, 2])
        with page_col1:
            compare_all_pages = st.toggle("比对所有页面", value=True)
        with page_col2:
            if not compare_all_pages:
                source_task_data = get_task_status(source_id)
                max_page = 1
                if source_task_data and source_task_data.get("result"):
                    max_page = len(source_task_data["result"].get("pages", [1]))
                page_number = st.number_input("指定页码", min_value=1, max_value=max_page, value=1)
            else:
                page_number = None

        with st.sidebar:
            st.divider()
            st.header("📊 比对任务列表")
            comp_data = list_comparisons()
            if comp_data and comp_data.get("comparisons"):
                comps = comp_data["comparisons"]
                st.info(f"共 {comp_data['total']} 个比对任务")

                for comp in reversed(comps[-10:]):
                    status_emoji = {
                        "pending": "⏳",
                        "processing": "⚙️",
                        "completed": "✅",
                        "failed": "❌",
                    }.get(comp["status"], "❓")

                    with st.expander(f"{status_emoji} {comp['comparison_id'][:8]}..."):
                        st.write(f"**状态**: {comp['status']}")
                        st.write(f"**源**: {comp['source_id'][:8]}...")
                        st.write(f"**目标**: {comp['target_id'][:8]}...")
                        st.write(f"**进度**: {comp['progress']*100:.0f}%")
                        if comp.get("message"):
                            st.caption(comp["message"])
                        st.write(f"**创建时间**: {comp['created_at']}")

                        if st.button("查看", key=f"view_comp_{comp['comparison_id']}"):
                            st.session_state["current_comparison_id"] = comp["comparison_id"]
            else:
                st.info("暂无比对任务")

        compare_btn = st.button(
            "🚀 开始比对",
            type="primary",
            disabled=source_id == target_id,
            use_container_width=True
        )

        if source_id == target_id:
            st.warning("请选择两个不同的任务进行比对")

        if compare_btn and source_id != target_id:
            with st.status("正在创建比对任务...", expanded=True) as status:
                result = create_comparison(source_id, target_id, page_number)
                if result and result.get("comparison_id"):
                    comparison_id = result["comparison_id"]
                    st.session_state["current_comparison_id"] = comparison_id
                    status.update(label=f"✅ 比对任务已创建: {comparison_id}", state="complete")
                    st.success(result["message"])
                    st.rerun()
                else:
                    status.update(label="❌ 创建比对任务失败", state="error")

        comparison_id = st.session_state.get("current_comparison_id", "")
        if not comparison_id:
            comparison_id = st.text_input(
                "或输入比对ID",
                value="",
                placeholder="输入比对ID或从侧边栏选择"
            )

        if comparison_id:
            comp_data = get_comparison_status(comparison_id)

            if not comp_data:
                st.error("比对任务不存在")
            else:
                status = comp_data["status"]
                progress = comp_data["progress"]

                st.subheader(f"比对状态: {status}")
                progress_bar = st.progress(progress)
                if comp_data.get("message"):
                    st.caption(comp_data["message"])

                if status == "pending":
                    st.info("比对任务排队中，请等待...")
                    time.sleep(2)
                    st.rerun()

                elif status == "processing":
                    st.info("比对进行中，正在刷新状态...")
                    time.sleep(2)
                    st.rerun()

                elif status == "completed" and comp_data.get("result"):
                    result = comp_data["result"]
                    page_diffs = result.get("page_diffs", [])
                    stats = result.get("stats", {})

                    st.divider()
                    st.markdown("### 📈 差异统计")

                    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5, stat_col6 = st.columns(6)
                    with stat_col1:
                        st.metric("新增", stats.get("added", 0))
                    with stat_col2:
                        st.metric("删除", stats.get("removed", 0))
                    with stat_col3:
                        st.metric("移动", stats.get("moved", 0))
                    with stat_col4:
                        st.metric("修改", stats.get("modified", 0))
                    with stat_col5:
                        st.metric("未变化", stats.get("unchanged", 0))
                    with stat_col6:
                        st.metric("总计", stats.get("total", 0))

                    st.divider()

                    if not page_diffs:
                        st.warning("没有比对结果")
                    else:
                        total_pages = len(page_diffs)
                        page_num = st.slider("选择页码", 1, total_pages, 1, key="compare_page_slider")

                        page_diff = page_diffs[page_num - 1]
                        diffs = page_diff.get("diffs", [])

                        show_unchanged = not st.toggle("仅显示差异区域", value=False, key="show_diff_only")

                        diff_type_filter = st.multiselect(
                            "筛选差异类型",
                            ["added", "removed", "moved", "modified", "unchanged"],
                            default=["added", "removed", "moved", "modified"] if not show_unchanged else ["added", "removed", "moved", "modified", "unchanged"],
                            format_func=lambda x: DIFF_TYPE_NAMES.get(x, x)
                        )

                        filtered_diffs = [d for d in diffs if d["type"] in diff_type_filter]

                        source_image = get_page_image(source_id, page_num)
                        target_image = get_page_image(target_id, page_num)

                        if source_image and target_image:
                            source_annotated = draw_diff_regions_on_image(
                                source_image, filtered_diffs, side="source", show_unchanged=show_unchanged
                            )
                            target_annotated = draw_diff_regions_on_image(
                                target_image, filtered_diffs, side="target", show_unchanged=show_unchanged
                            )

                            img_col1, img_col2 = st.columns(2)
                            with img_col1:
                                st.markdown(f"**源版本 (Source)** - 第 {page_num} 页")
                                st.caption(f"红色框=删除区域, 黄色框=移动区域")
                                st.image(source_annotated, use_container_width=True)

                            with img_col2:
                                st.markdown(f"**目标版本 (Target)** - 第 {page_num} 页")
                                st.caption(f"绿色框=新增区域, 黄色框=移动区域")
                                st.image(target_annotated, use_container_width=True)

                            if any(d["type"] == "moved" for d in filtered_diffs):
                                st.divider()
                                st.markdown("### 🔄 移动区域连接示意")
                                st.caption("虚线连接移动区域的旧位置和新位置")
                                moved_diffs = [d for d in filtered_diffs if d["type"] == "moved"]
                                combined_img = draw_moved_connections(source_image, target_image, moved_diffs)
                                st.image(combined_img, use_container_width=True)

                        st.divider()
                        st.markdown("### 📋 差异详情列表")

                        type_order = {"added": 0, "removed": 1, "moved": 2, "modified": 3, "unchanged": 4}
                        sorted_diffs = sorted(filtered_diffs, key=lambda d: type_order.get(d["type"], 99))

                        for idx, diff in enumerate(sorted_diffs):
                            diff_type = diff["type"]
                            color = DIFF_TYPE_COLORS.get(diff_type, (128, 128, 128))
                            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                            with st.expander(
                                f"{idx + 1}. [{DIFF_TYPE_NAMES.get(diff_type, diff_type)}] "
                                f"区域: {diff.get('source_region_id') or diff.get('target_region_id', 'N/A')[:8]}...",
                                expanded=diff_type != "unchanged"
                            ):
                                st.markdown(
                                    f"<div style='background-color:{color_hex}20;padding:10px;border-radius:5px;"
                                    f"border-left:4px solid {color_hex}'>"
                                    f"<strong>差异类型:</strong> {DIFF_TYPE_NAMES.get(diff_type, diff_type)}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

                                detail_col1, detail_col2 = st.columns(2)
                                with detail_col1:
                                    st.write(f"**页码**: {diff['page_number']}")
                                    if diff.get("source_region_id"):
                                        st.write(f"**源区域ID**: {diff['source_region_id']}")
                                    if diff.get("target_region_id"):
                                        st.write(f"**目标区域ID**: {diff['target_region_id']}")
                                    if diff.get("iou") is not None:
                                        st.write(f"**IoU匹配度**: {diff['iou']:.4f}")

                                with detail_col2:
                                    if diff.get("source_region"):
                                        src_region = diff["source_region"]
                                        st.write(f"**源区域类型**: {REGION_NAMES.get(src_region['type'], src_region['type'])}")
                                        bbox = src_region["bbox"]
                                        st.write(f"**源位置**: x={bbox['x']:.4f}, y={bbox['y']:.4f}, w={bbox['width']:.4f}, h={bbox['height']:.4f}")
                                    if diff.get("target_region"):
                                        tgt_region = diff["target_region"]
                                        st.write(f"**目标区域类型**: {REGION_NAMES.get(tgt_region['type'], tgt_region['type'])}")
                                        bbox = tgt_region["bbox"]
                                        st.write(f"**目标位置**: x={bbox['x']:.4f}, y={bbox['y']:.4f}, w={bbox['width']:.4f}, h={bbox['height']:.4f}")

                                if diff.get("displacement"):
                                    disp = diff["displacement"]
                                    st.info(
                                        f"**位移向量**: dx={disp['dx']:.4f}, dy={disp['dy']:.4f} "
                                        f"(水平偏移: {disp['dx']*100:.1f}%, 垂直偏移: {disp['dy']*100:.1f}%)"
                                    )

                                if diff.get("content_summary"):
                                    st.markdown("#### 📝 内容变化摘要")
                                    st.info(diff["content_summary"])

                                    if diff.get("source_region") and diff["source_region"].get("text"):
                                        with st.expander("查看源文本"):
                                            st.text(diff["source_region"]["text"])
                                    if diff.get("target_region") and diff["target_region"].get("text"):
                                        with st.expander("查看目标文本"):
                                            st.text(diff["target_region"]["text"])

                        st.divider()
                        st.markdown("### 📁 导出比对结果")

                        export_col1, export_col2 = st.columns(2)
                        with export_col1:
                            if st.button("📄 导出 JSON 报告", use_container_width=True):
                                export_resp = export_comparison(comparison_id)
                                if export_resp:
                                    st.download_button(
                                        "⬇️ 下载 JSON 报告",
                                        export_resp.content,
                                        file_name=f"comparison_{comparison_id[:8]}.json",
                                        mime="application/json"
                                    )

                        with export_col2:
                            if st.button("👁️ 预览 JSON 数据", use_container_width=True):
                                st.json(result)

                elif status == "failed":
                    st.error(f"比对失败: {comp_data.get('error', '未知错误')}")

with tab4:
    st.header("使用说明")

    st.markdown("""
    ### 🎯 系统功能

    这是一个文档布局分析与版面区域分割系统，可以自动识别文档中的不同区域类型：

    | 区域类型 | 说明 |
    |---------|------|
    | 📝 正文 | 普通文本段落 |
    | 📌 一级/二级/三级标题 | 各级标题，区分层级 |
    | 📊 表格 | 整个表格区域，支持结构化分析 |
    | 🖼️ 图片/图表 | 插图区域 |
    | 🔖 图注/表注 | 图表下方的说明文字 |
    | 📑 页眉/页脚 | 页面顶部/底部的重复内容 |
    | 📐 数学公式 | 行内公式和独立公式块 |
    | 📋 列表 | 有序和无序列表 |

    ---

    ### 🔧 核心技术

    - **ONNX模型推理**: 预训练版面检测模型，支持批量页面并行处理
    - **XY-Cut递归分割**: 智能推断阅读顺序，支持多栏文档
    - **Hough变换**: 倾斜校正和表格线条检测
    - **OCR文字识别**: 基于Tesseract的中英文识别
    - **表格结构识别**: 支持有线/无线表格，自动检测合并单元格

    ---

    ### 📡 API接口

    **文档分析接口:**
    - `POST /analyze` - 上传文档进行分析（异步）
    - `GET /tasks/{id}` - 查询任务状态和结果
    - `GET /tasks/{id}/pages/{page}` - 获取单页详细结果
    - `GET /tasks/{id}/export?format=json|hocr|alto` - 导出分析结果
    - `DELETE /tasks/{id}` - 删除任务

    **版面比对接口:**
    - `POST /compare` - 创建比对任务（异步），请求体: `{source_id, target_id, page_number?}`
    - `GET /comparisons` - 列出所有比对任务
    - `GET /comparisons/{id}` - 查询比对状态和结果
    - `GET /comparisons/{id}/export` - 导出结构化差异报告（JSON）

    ---

    ### 💡 使用技巧

    1. **扫描件处理**: 系统会自动进行倾斜校正和分辨率归一化
    2. **多栏文档**: XY-Cut算法会自动处理双栏、三栏布局
    3. **表格识别**: 点击表格区域可查看结构化的表格内容
    4. **阅读顺序**: 区域上标注的数字即为阅读顺序

    ---

    ### 🎨 区域颜色说明
    """)

    st.markdown("**区域类型颜色:**")
    color_cols = st.columns(4)
    for i, (rtype, color) in enumerate(REGION_COLORS.items()):
        with color_cols[i % 4]:
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            st.markdown(
                f"<div style='background-color:{color_hex};padding:10px;border-radius:5px;color:black'>"
                f"{REGION_NAMES.get(rtype, rtype)}</div>",
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown("**差异类型颜色 (版面比对):**")
    diff_color_cols = st.columns(5)
    for i, (dtype, color) in enumerate(DIFF_TYPE_COLORS.items()):
        with diff_color_cols[i % 5]:
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            st.markdown(
                f"<div style='background-color:{color_hex};padding:10px;border-radius:5px;color:white'>"
                f"{DIFF_TYPE_NAMES.get(dtype, dtype)}</div>",
                unsafe_allow_html=True
            )

    st.divider()
    st.caption("版本: 1.0.0 | 基于 Flask + Streamlit + ONNX Runtime")
