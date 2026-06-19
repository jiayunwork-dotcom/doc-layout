from typing import List, Optional, Dict
from schemas.models import (
    Page,
    Region,
    RegionType,
    HierarchyNode,
)
import uuid


class HierarchyExtractor:
    def __init__(self):
        self.title_levels = {
            RegionType.TITLE_H1: 1,
            RegionType.TITLE_H2: 2,
            RegionType.TITLE_H3: 3,
        }

    def extract_hierarchy(self, pages: List[Page]) -> Optional[HierarchyNode]:
        all_titles = []
        all_content = []

        for page in pages:
            sorted_regions = sorted(
                page.regions,
                key=lambda r: r.reading_order if r.reading_order is not None else 9999
            )

            for region in sorted_regions:
                if region.type in self.title_levels:
                    all_titles.append((page.page_number, region))
                elif region.type in {RegionType.TEXT, RegionType.LIST, RegionType.FIGURE,
                                     RegionType.TABLE, RegionType.FORMULA}:
                    all_content.append((page.page_number, region))

        if not all_titles and not all_content:
            return None

        root = HierarchyNode(
            node_id=str(uuid.uuid4()),
            title="Document Root",
            level=0,
        )

        self._build_hierarchy(root, all_titles, all_content)
        return root

    def _build_hierarchy(self, root: HierarchyNode,
                         titles: List[tuple],
                         content: List[tuple]) -> None:
        node_stack = [(root, 0)]
        content_idx = 0

        for page_num, title_region in titles:
            level = self.title_levels.get(title_region.type, 1)

            while node_stack and node_stack[-1][1] >= level:
                node_stack.pop()

            if not node_stack:
                parent = root
            else:
                parent = node_stack[-1][0]

            title_node = HierarchyNode(
                node_id=str(uuid.uuid4()),
                title=title_region.text or f"{title_region.type.value}",
                level=level,
                region_ids=[title_region.id],
            )
            parent.children.append(title_node)
            node_stack.append((title_node, level))

            while content_idx < len(content):
                content_page, content_region = content[content_idx]

                if (content_page > page_num or
                    (content_page == page_num and
                     title_region.reading_order is not None and
                     content_region.reading_order is not None and
                     content_region.reading_order > title_region.reading_order)):
                    break

                next_title_page, next_title_region = titles[titles.index((page_num, title_region)) + 1] \
                    if titles.index((page_num, title_region)) + 1 < len(titles) else (float("inf"), None)

                if content_page > next_title_page:
                    break
                if (content_page == next_title_page and
                    next_title_region is not None and
                    next_title_region.reading_order is not None and
                    content_region.reading_order is not None and
                    content_region.reading_order > next_title_region.reading_order):
                    break

                title_node.region_ids.append(content_region.id)
                content_region.parent_id = title_node.node_id
                content_idx += 1

        while content_idx < len(content):
            content_page, content_region = content[content_idx]
            root.region_ids.append(content_region.id)
            content_region.parent_id = root.node_id
            content_idx += 1

    def get_toc(self, hierarchy: HierarchyNode) -> List[Dict]:
        toc = []
        self._traverse_toc(hierarchy, toc)
        return toc

    def _traverse_toc(self, node: HierarchyNode, toc: List[Dict], depth: int = 0) -> None:
        if node.level > 0:
            toc.append({
                "title": node.title,
                "level": node.level,
                "depth": depth,
                "region_ids": node.region_ids,
                "node_id": node.node_id,
            })

        for child in node.children:
            self._traverse_toc(child, toc, depth + 1)

    def get_hierarchy_flat(self, hierarchy: HierarchyNode) -> List[Dict]:
        flat = []
        self._flatten(hierarchy, flat)
        return flat

    def _flatten(self, node: HierarchyNode, flat: List[Dict], path: str = "") -> None:
        current_path = f"{path}/{node.title}" if path else node.title

        flat.append({
            "node_id": node.node_id,
            "title": node.title,
            "level": node.level,
            "region_ids": node.region_ids,
            "path": current_path,
            "child_count": len(node.children),
        })

        for child in node.children:
            self._flatten(child, flat, current_path)

    def get_region_hierarchy_map(self, pages: List[Page],
                                  hierarchy: HierarchyNode) -> Dict[str, List[str]]:
        region_map = {}

        def traverse(node, ancestors):
            current_ancestors = ancestors + [node.node_id]

            for region_id in node.region_ids:
                region_map[region_id] = current_ancestors

            for child in node.children:
                traverse(child, current_ancestors)

        traverse(hierarchy, [])
        return region_map

    def extract_list_hierarchy(self, regions: List[Region]) -> Dict[str, int]:
        list_regions = [r for r in regions if r.type == RegionType.LIST]
        hierarchy = {}

        for list_region in list_regions:
            indent = list_region.bbox.x
            if indent < 0.1:
                level = 1
            elif indent < 0.15:
                level = 2
            else:
                level = 3
            hierarchy[list_region.id] = level

        return hierarchy
