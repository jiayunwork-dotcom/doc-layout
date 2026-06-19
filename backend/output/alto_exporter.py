from typing import List
from datetime import datetime
from schemas.models import AnalysisResult, Page, Region, RegionType, TableStructure, TableCell
from lxml import etree


class ALTOExporter:
    NAMESPACE = "http://www.loc.gov/standards/alto/ns-v4#"
    XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"
    SCHEMA_LOCATION = "http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-3.xsd"

    TYPE_MAP = {
        RegionType.TEXT: "TextBlock",
        RegionType.TITLE_H1: "TextBlock",
        RegionType.TITLE_H2: "TextBlock",
        RegionType.TITLE_H3: "TextBlock",
        RegionType.TABLE: "Table",
        RegionType.FIGURE: "Illustration",
        RegionType.CAPTION: "TextBlock",
        RegionType.HEADER: "Header",
        RegionType.FOOTER: "Footer",
        RegionType.SIDEBAR: "TextBlock",
        RegionType.FORMULA: "MathsBlock",
        RegionType.FORMULA_INLINE: "MathsBlock",
        RegionType.LIST: "TextBlock",
    }

    @staticmethod
    def export(result: AnalysisResult) -> str:
        alto = ALTOExporter._build_alto(result)
        return etree.tostring(alto, encoding="unicode", pretty_print=True)

    @staticmethod
    def _build_alto(result: AnalysisResult) -> etree._Element:
        nsmap = {
            None: ALTOExporter.NAMESPACE,
            "xsi": ALTOExporter.XSI_NAMESPACE,
        }

        alto = etree.Element("alto", nsmap=nsmap)
        alto.set(f"{{{ALTOExporter.XSI_NAMESPACE}}}schemaLocation", ALTOExporter.SCHEMA_LOCATION)

        description = etree.SubElement(alto, "Description")

        measurement_unit = etree.SubElement(description, "MeasurementUnit")
        measurement_unit.text = "pixel"

        source_image_information = etree.SubElement(description, "sourceImageInformation")
        file_name = etree.SubElement(source_image_information, "fileName")
        file_name.text = result.metadata.filename

        processing = etree.SubElement(description, "processing")
        processing.set("ID", "DocLayoutAnalysis")
        processing.set("processingDateTime", datetime.now().isoformat())

        processing_step = etree.SubElement(processing, "processingStep")
        processing_step.set("processingStepDescription", "Document Layout Analysis")
        processing_step.set("processingStepSettings", "v1.0")

        software = etree.SubElement(processing_step, "processingSoftware")
        software_name = etree.SubElement(software, "softwareName")
        software_name.text = "DocLayout Analysis"
        software_version = etree.SubElement(software, "softwareVersion")
        software_version.text = "1.0"

        layout = etree.SubElement(alto, "Layout")

        for page in result.pages:
            page_elem = ALTOExporter._build_page(page)
            layout.append(page_elem)

        return alto

    @staticmethod
    def _build_page(page: Page) -> etree._Element:
        page_elem = etree.Element("Page")
        page_elem.set("ID", f"page_{page.page_number}")
        page_elem.set("PHYSICAL_IMG_NR", str(page.page_number))
        page_elem.set("HEIGHT", str(page.height))
        page_elem.set("WIDTH", str(page.width))
        page_elem.set("DPI", str(page.dpi))

        print_space = etree.SubElement(page_elem, "PrintSpace")
        print_space.set("HPOS", "0")
        print_space.set("VPOS", "0")
        print_space.set("WIDTH", str(page.width))
        print_space.set("HEIGHT", str(page.height))

        sorted_regions = sorted(
            page.regions,
            key=lambda r: r.reading_order if r.reading_order is not None else 9999
        )

        for region in sorted_regions:
            block_elem = ALTOExporter._build_block(region, page)
            if block_elem is not None:
                print_space.append(block_elem)

        return page_elem

    @staticmethod
    def _build_block(region: Region, page: Page) -> etree._Element:
        block_type = ALTOExporter.TYPE_MAP.get(region.type, "TextBlock")

        if block_type == "Table":
            return ALTOExporter._build_table_block(region, page)

        block = etree.Element(block_type)
        block.set("ID", f"block_{region.id}")

        bbox = region.bbox.to_absolute(page.width, page.height)
        block.set("HPOS", str(bbox[0]))
        block.set("VPOS", str(bbox[1]))
        block.set("WIDTH", str(bbox[2] - bbox[0]))
        block.set("HEIGHT", str(bbox[3] - bbox[1]))

        if region.reading_order is not None:
            block.set("ORDER", str(region.reading_order))

        if region.confidence is not None:
            block.set("CS", f"{region.confidence:.4f}")

        block.set("TYPE", region.type.value)

        if region.type in {RegionType.TITLE_H1, RegionType.TITLE_H2, RegionType.TITLE_H3}:
            block.set("TYPE", f"heading-{region.type.value[-1]}")

        if region.text:
            text_line = etree.SubElement(block, "TextLine")
            text_line.set("HPOS", str(bbox[0]))
            text_line.set("VPOS", str(bbox[1]))
            text_line.set("WIDTH", str(bbox[2] - bbox[0]))
            text_line.set("HEIGHT", str(bbox[3] - bbox[1]))

            string = etree.SubElement(text_line, "String")
            string.set("CONTENT", region.text)
            string.set("HPOS", str(bbox[0]))
            string.set("VPOS", str(bbox[1]))
            string.set("WIDTH", str(bbox[2] - bbox[0]))
            string.set("HEIGHT", str(bbox[3] - bbox[1]))

        if block_type == "Illustration":
            graph_element = etree.SubElement(block, "GraphicalElement")
            graph_element.set("ID", f"graphic_{region.id}")
            graph_element.set("HPOS", str(bbox[0]))
            graph_element.set("VPOS", str(bbox[1]))
            graph_element.set("WIDTH", str(bbox[2] - bbox[0]))
            graph_element.set("HEIGHT", str(bbox[3] - bbox[1]))

        for child_id in region.children:
            child_region = next((r for r in page.regions if r.id == child_id), None)
            if child_region:
                child_block = ALTOExporter._build_block(child_region, page)
                if child_block is not None:
                    block.append(child_block)

        return block

    @staticmethod
    def _build_table_block(region: Region, page: Page) -> etree._Element:
        table = etree.Element("Table")
        table.set("ID", f"table_{region.id}")

        bbox = region.bbox.to_absolute(page.width, page.height)
        table.set("HPOS", str(bbox[0]))
        table.set("VPOS", str(bbox[1]))
        table.set("WIDTH", str(bbox[2] - bbox[0]))
        table.set("HEIGHT", str(bbox[3] - bbox[1]))

        if region.reading_order is not None:
            table.set("ORDER", str(region.reading_order))

        if region.confidence is not None:
            table.set("CS", f"{region.confidence:.4f}")

        if region.table_structure and region.table_structure.cells:
            table.set("ROWS", str(region.table_structure.rows))
            table.set("COLS", str(region.table_structure.cols))

            for cell in region.table_structure.cells:
                cell_elem = ALTOExporter._build_table_cell(cell, page)
                table.append(cell_elem)

        return table

    @staticmethod
    def _build_table_cell(cell: TableCell, page: Page) -> etree._Element:
        cell_elem = etree.Element("TableCell")
        cell_elem.set("ID", f"cell_{cell.row_index}_{cell.col_index}")
        cell_elem.set("ROW", str(cell.row_index))
        cell_elem.set("COL", str(cell.col_index))
        cell_elem.set("ROW_SPAN", str(cell.row_span))
        cell_elem.set("COL_SPAN", str(cell.col_span))

        bbox = cell.bbox.to_absolute(page.width, page.height)
        cell_elem.set("HPOS", str(bbox[0]))
        cell_elem.set("VPOS", str(bbox[1]))
        cell_elem.set("WIDTH", str(bbox[2] - bbox[0]))
        cell_elem.set("HEIGHT", str(bbox[3] - bbox[1]))

        if cell.text:
            text_line = etree.SubElement(cell_elem, "TextLine")
            text_line.set("HPOS", str(bbox[0]))
            text_line.set("VPOS", str(bbox[1]))
            text_line.set("WIDTH", str(bbox[2] - bbox[0]))
            text_line.set("HEIGHT", str(bbox[3] - bbox[1]))

            string = etree.SubElement(text_line, "String")
            string.set("CONTENT", cell.text)
            string.set("HPOS", str(bbox[0]))
            string.set("VPOS", str(bbox[1]))
            string.set("WIDTH", str(bbox[2] - bbox[0]))
            string.set("HEIGHT", str(bbox[3] - bbox[1]))

        return cell_elem

    @staticmethod
    def save(result: AnalysisResult, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(ALTOExporter.export(result))
