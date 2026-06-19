from typing import List
from schemas.models import AnalysisResult, Page, Region, RegionType
from lxml import etree


class HOCRExporter:
    REGION_CLASS_MAP = {
        RegionType.TEXT: "ocr_par",
        RegionType.TITLE_H1: "ocr_title",
        RegionType.TITLE_H2: "ocr_title",
        RegionType.TITLE_H3: "ocr_title",
        RegionType.TABLE: "ocr_table",
        RegionType.FIGURE: "ocr_figure",
        RegionType.CAPTION: "ocr_caption",
        RegionType.HEADER: "ocr_header",
        RegionType.FOOTER: "ocr_footer",
        RegionType.SIDEBAR: "ocr_sidebar",
        RegionType.FORMULA: "ocr_formula",
        RegionType.FORMULA_INLINE: "ocr_formula",
        RegionType.LIST: "ocr_list",
    }

    @staticmethod
    def export(result: AnalysisResult) -> str:
        html = HOCRExporter._build_html(result)
        return etree.tostring(html, encoding="unicode", pretty_print=True)

    @staticmethod
    def _build_html(result: AnalysisResult) -> etree._Element:
        html = etree.Element("html", xmlns="http://www.w3.org/1999/xhtml")
        head = etree.SubElement(html, "head")

        meta = etree.SubElement(head, "meta")
        meta.set("name", "ocr-system")
        meta.set("content", "DocLayout Analysis v1.0")

        meta2 = etree.SubElement(head, "meta")
        meta2.set("name", "ocr-capabilities")
        meta2.set("content", "ocr_page ocr_carea ocr_par ocr_line ocrx_word")

        title = etree.SubElement(head, "title")
        title.text = f"hOCR - {result.metadata.filename}"

        body = etree.SubElement(html, "body")

        for page in result.pages:
            page_elem = HOCRExporter._build_page(page)
            body.append(page_elem)

        return html

    @staticmethod
    def _build_page(page: Page) -> etree._Element:
        div = etree.Element("div")
        div.set("class", "ocr_page")

        bbox_str = f"bbox 0 0 {page.width} {page.height}"
        div.set("title", f"{bbox_str}; page_number {page.page_number}")

        sorted_regions = sorted(
            page.regions,
            key=lambda r: r.reading_order if r.reading_order is not None else 9999
        )

        for region in sorted_regions:
            region_elem = HOCRExporter._build_region(region, page)
            div.append(region_elem)

        return div

    @staticmethod
    def _build_region(region: Region, page: Page) -> etree._Element:
        div = etree.Element("div")

        region_class = HOCRExporter.REGION_CLASS_MAP.get(region.type, "ocr_carea")
        div.set("class", f"{region_class} {region.type.value}")

        bbox = region.bbox.to_absolute(page.width, page.height)
        bbox_str = f"bbox {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"

        title_parts = [bbox_str]
        title_parts.append(f"confidence {region.confidence:.4f}")

        if region.reading_order is not None:
            title_parts.append(f"reading_order {region.reading_order}")

        div.set("title", "; ".join(title_parts))
        div.set("id", f"region_{region.id}")

        if region.text:
            p = etree.SubElement(div, "p")
            p.text = region.text

        if region.table_structure and region.table_structure.cells:
            table = etree.SubElement(div, "table")
            table.set("class", "ocr_table_structure")

            grid = {}
            for cell in region.table_structure.cells:
                for ri in range(cell.row_span):
                    for ci in range(cell.col_span):
                        grid[(cell.row_index + ri, cell.col_index + ci)] = cell

            for row_idx in range(region.table_structure.rows):
                tr = etree.SubElement(table, "tr")
                for col_idx in range(region.table_structure.cols):
                    cell = grid.get((row_idx, col_idx))
                    if cell and cell.row_index == row_idx and cell.col_index == col_idx:
                        td = etree.SubElement(tr, "td")
                        td.set("rowspan", str(cell.row_span))
                        td.set("colspan", str(cell.col_span))

                        cell_bbox = cell.bbox.to_absolute(page.width, page.height)
                        td.set("title", f"bbox {cell_bbox[0]} {cell_bbox[1]} {cell_bbox[2]} {cell_bbox[3]}")

                        if cell.text:
                            td.text = cell.text

        for child_id in region.children:
            child_region = next((r for r in page.regions if r.id == child_id), None)
            if child_region:
                child_elem = HOCRExporter._build_region(child_region, page)
                div.append(child_elem)

        return div

    @staticmethod
    def save(result: AnalysisResult, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(HOCRExporter.export(result))
