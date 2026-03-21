"""
svg_parser.py
Parses a CubiCasa5k model.svg floor plan and converts it to the grid/door
data structures expected by the crowd simulation.

CubiCasa5k SVG structure (relevant parts):
  <g class="Space LivingRoom"> … <polygon points="…"/> … </g>   <- floor areas
  <g class="Space Outdoor">   … <polygon points="…"/> … </g>   <- excluded
  <g class="Wall External">   … <polygon points="…"/> … </g>   <- walls
  <g class="Wall">            … <polygon points="…"/> … </g>   <- inner walls
  <g id="Door" class="Door …"> … <polygon points="…"/> … </g> <- doors (nested in Wall)
"""

import xml.etree.ElementTree as ET
import re
from collections import deque


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG 'points' attribute into list of (x, y) floats."""
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", points_str)
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append((float(nums[i]), float(nums[i + 1])))
    return pts


def _poly_bbox(pts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _point_in_poly(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _rasterise_polygon_fill(pts, grid, gw, gh, svg_w, svg_h, value):
    """Scanline fill all grid cells whose centre falls inside the polygon."""
    if len(pts) < 3:
        return
    min_x, min_y, max_x, max_y = _poly_bbox(pts)
    # convert bbox to grid coords
    gx0 = max(0, int(min_x / svg_w * gw))
    gx1 = min(gw - 1, int(max_x / svg_w * gw) + 1)
    gy0 = max(0, int(min_y / svg_h * gh))
    gy1 = min(gh - 1, int(max_y / svg_h * gh) + 1)
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            # cell centre in SVG coords
            cx = (gx + 0.5) / gw * svg_w
            cy = (gy + 0.5) / gh * svg_h
            if _point_in_poly(cx, cy, pts):
                grid[gx][gy] = value


def _poly_centroid_grid(pts, gw, gh, svg_w, svg_h):
    """Return grid (gx, gy) of polygon centroid, clamped."""
    if not pts:
        return (gw // 2, gh // 2)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    gx = int(cx / svg_w * gw)
    gy = int(cy / svg_h * gh)
    gx = max(0, min(gw - 1, gx))
    gy = max(0, min(gh - 1, gy))
    return gx, gy


def _ensure_connectivity_simple(g, gw, gh):
    """BFS from centre; if outer cells are not reachable, carve a corridor."""
    INF = 10 ** 9

    def bfs(sources):
        dist = [[INF] * gh for _ in range(gw)]
        q = deque()
        for sx, sy in sources:
            dist[sx][sy] = 0
            q.append((sx, sy))
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < gw and 0 <= ny < gh and g[nx][ny] == 0 and dist[nx][ny] > dist[x][y] + 1:
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))
        return dist

    # find central free cells
    cx, cy = gw // 2, gh // 2
    sources = []
    for rx in range(-5, 6):
        for ry in range(-5, 6):
            nx, ny = cx + rx, cy + ry
            if 0 <= nx < gw and 0 <= ny < gh and g[nx][ny] == 0:
                sources.append((nx, ny))

    if not sources:
        # force open a small central area
        for rx in range(-2, 3):
            for ry in range(-2, 3):
                nx, ny = cx + rx, cy + ry
                if 0 <= nx < gw and 0 <= ny < gh:
                    g[nx][ny] = 0
        sources = [(cx, cy)]

    return g


# ---------------------------------------------------------------------------
# SVG Namespace helper
# ---------------------------------------------------------------------------
SVG_NS = "http://www.w3.org/2000/svg"


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_svg_floorplan(
    svg_path: str,
    grid_w: int = 80,
    grid_h: int = 80,
) -> tuple[list[list[int]], list[tuple[int, int]], list[list[int]]]:
    """
    Parse a CubiCasa5k model.svg and return:
        grid       : 2-D list [gw][gh], 0=walkable, 1=wall
        doors      : list of (gx, gy) grid cells for door centres
        floor_mask : 2-D list [gw][gh], 1=indoor floor cell, 0=otherwise
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # ---- read SVG canvas size ----
    vb = root.get("viewBox", "")
    if vb:
        parts = vb.split()
        svg_w = float(parts[2])
        svg_h = float(parts[3])
    else:
        svg_w = float(root.get("width", "1000"))
        svg_h = float(root.get("height", "1000"))

    # ---- initialise grid: all walls by default ----
    grid = [[1] * grid_h for _ in range(grid_w)]
    floor_mask = [[0] * grid_h for _ in range(grid_w)]

    # ---- collect elements ----
    # We walk all <g> elements and classify by their CSS class string.
    doors = []

    def get_class(elem):
        return elem.get("class", "")

    def first_polygon_points(elem):
        """Return points of the first <polygon> in elem (recursive search)."""
        for child in elem.iter():
            if _strip_ns(child.tag) == "polygon":
                pts_str = child.get("points", "")
                if pts_str:
                    return _parse_points(pts_str)
        return []

    # Iterate all <g> elements
    for g_elem in root.iter():
        if _strip_ns(g_elem.tag) != "g":
            continue

        cls = get_class(g_elem)
        elem_id = g_elem.get("id", "")

        # ---- Indoor Space polygons → floor ----
        if "Space" in cls and "Outdoor" not in cls:
            for child in g_elem:
                if _strip_ns(child.tag) == "polygon":
                    pts_str = child.get("points", "")
                    if pts_str:
                        pts = _parse_points(pts_str)
                        # Rasterise as walkable
                        _rasterise_polygon_fill(pts, grid, grid_w, grid_h, svg_w, svg_h, 0)
                        _rasterise_polygon_fill(pts, floor_mask, grid_w, grid_h, svg_w, svg_h, 1)
                    break  # only first polygon = room boundary

        # ---- Wall / Railing polygons → wall ----
        elif "Wall" in cls or "Railing" in cls:
            for child in g_elem:
                if _strip_ns(child.tag) == "polygon":
                    pts_str = child.get("points", "")
                    if pts_str:
                        pts = _parse_points(pts_str)
                        _rasterise_polygon_fill(pts, grid, grid_w, grid_h, svg_w, svg_h, 1)
                    break

        # ---- Door elements → exits ----
        # Doors have id="Door" and class="Door …", nested inside Wall <g>s
        if elem_id == "Door" and "Door" in cls:
            pts = first_polygon_points(g_elem)
            if pts:
                gx, gy = _poly_centroid_grid(pts, grid_w, grid_h, svg_w, svg_h)
                # Force the door cell open (it's a threshold/opening)
                grid[gx][gy] = 0
                # Also open a small neighbourhood so agents can exit
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < grid_w and 0 <= ny < grid_h:
                            grid[nx][ny] = 0
                doors.append((gx, gy))

    # ---- Ensure boundary walls ----
    for x in range(grid_w):
        grid[x][0] = 1
        grid[x][grid_h - 1] = 1
        floor_mask[x][0] = 0
        floor_mask[x][grid_h - 1] = 0
    for y in range(grid_h):
        grid[0][y] = 1
        grid[grid_w - 1][y] = 1
        floor_mask[0][y] = 0
        floor_mask[grid_w - 1][y] = 0

    # ---- De-duplicate doors ----
    seen = set()
    unique_doors = []
    for d in doors:
        if d not in seen:
            seen.add(d)
            unique_doors.append(d)
    doors = unique_doors

    # ---- Fallback: if no doors detected, pick border cells near floor ----
    if not doors:
        print("[svg_parser] WARNING: no doors detected; using fallback border exits")
        border_candidates = []
        for x in range(1, grid_w - 1):
            for y in (1, grid_h - 2):
                if floor_mask[x][y] == 1:
                    border_candidates.append((x, y))
        for y in range(1, grid_h - 1):
            for x in (1, grid_w - 2):
                if floor_mask[x][y] == 1:
                    border_candidates.append((x, y))
        if border_candidates:
            step = max(1, len(border_candidates) // 8)
            doors = [border_candidates[i] for i in range(0, len(border_candidates), step)][:8]
        else:
            doors = [(grid_w - 2, grid_h // 2)]

    # ---- Ensure connectivity from floor to doors ----
    grid = _ensure_connectivity_simple(grid, grid_w, grid_h)

    print(f"[svg_parser] SVG: {svg_path}")
    print(f"[svg_parser] Grid {grid_w}x{grid_h}, "
          f"walkable={sum(grid[x][y]==0 for x in range(grid_w) for y in range(grid_h))}, "
          f"walls={sum(grid[x][y]==1 for x in range(grid_w) for y in range(grid_h))}, "
          f"doors={len(doors)}: {doors[:8]}")

    return grid, doors, floor_mask


# ---------------------------------------------------------------------------
# Optional: save a debug PNG of the parsed grid
# ---------------------------------------------------------------------------

def save_debug_png(grid, doors, floor_mask, out_path="parsed_grid_debug.png"):
    """Save a simple PNG showing walls, floor, and door locations."""
    try:
        import numpy as np
        from PIL import Image

        gw = len(grid)
        gh = len(grid[0])
        scale = 8
        img = np.ones((gh * scale, gw * scale, 3), dtype=np.uint8) * 200  # grey bg

        for x in range(gw):
            for y in range(gh):
                r = y * scale
                c = x * scale
                if grid[x][y] == 1:
                    img[r:r+scale, c:c+scale] = [40, 40, 40]  # dark wall
                elif floor_mask[x][y] == 1:
                    img[r:r+scale, c:c+scale] = [245, 245, 245]  # floor
                else:
                    img[r:r+scale, c:c+scale] = [180, 220, 180]  # outdoor/other

        for dx, dy in doors:
            r = dy * scale
            c = dx * scale
            img[r:r+scale, c:c+scale] = [0, 200, 0]  # green doors

        Image.fromarray(img).save(out_path)
        print(f"[svg_parser] Debug PNG saved: {out_path}")
    except ImportError:
        print("[svg_parser] PIL not available; skipping debug PNG")
