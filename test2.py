"""
batch_parse_cubicasa.py

Hard-coded input/output:
  INPUT_BASE  = "cubicasa5k/high_quality"
  OUTPUT_BASE = "processed_data"

Usage:
  python batch_parse_cubicasa.py
"""

import os
import cv2
import numpy as np
from xml.dom import minidom

# Optional: better path sampling for <path d="..."> elements
try:
    import svgpathtools as spt
except Exception:
    spt = None

INPUT_BASE = "cubicasa5k/high_quality"
OUTPUT_BASE = "processed_data"
DEFAULT_CANV = 1024


def points_from_str(points_str):
    toks = points_str.replace(",", " ").split()
    pts = []
    it = iter(toks)
    # zip(it, it) pairs consecutive tokens; if odd count, last ignored
    for x, y in zip(it, it):
        try:
            pts.append((float(x), float(y)))
        except Exception:
            continue
    return pts


def sample_path_d(d_attr, n_samples=200):
    """Return list of (x,y) by sampling path. Requires svgpathtools."""
    if spt is None or not d_attr:
        return []
    try:
        path = spt.parse_path(d_attr)
        if path.length() == 0:
            return []
        pts = []
        # sample by normalized arclength
        L = path.length()
        for i in range(n_samples + 1):
            t = i / float(n_samples)
            pt = path.point(t * path.length() / path.length() if False else t)
            pts.append((pt.real, pt.imag))
        return pts
    except Exception:
        return []


def get_ancestor_group_label(node):
    """Walk up ancestors to find a group id/class string, return concatenated label (lower)."""
    parts = []
    cur = node
    while cur is not None and cur.nodeType == cur.ELEMENT_NODE:
        if cur.hasAttribute and (cur.hasAttribute("id") or cur.hasAttribute("class")):
            gid = cur.getAttribute("id") if cur.hasAttribute("id") else ""
            gcls = cur.getAttribute("class") if cur.hasAttribute("class") else ""
            combined = (gid + " " + gcls).strip()
            if combined:
                parts.append(combined)
        cur = cur.parentNode
    return " ".join(reversed(parts)).lower()


def classify_element(el, style_fill, group_label):
    """
    Decide whether an element is wall / door / room / other using group_label and fill color.
    Returns one of: "wall", "door", "room", None (None -> default to room)
    """
    fill = (style_fill or "").strip().lower()
    g = (group_label or "").lower()

    # group-based heuristics
    if "door" in g or "door" in fill:
        return "door"
    if "wall" in g or "wall" in fill:
        return "wall"
    if "room" in g or "floor" in g or "area" in g:
        return "room"

    # walls -> black-ish; doors -> red-ish; rooms -?>light gray
    if fill.startswith("#"):
        if fill in ("#000000", "#010101", "#1a1a1a"):
            return "wall"
        if fill in ("#ff0000", "#ff7f00", "#ff0000ff"):
            return "door"
        if fill in ("#f5deb3", "#bebebe", "#e5e5e5", "#dcdcdc", "#ffffff"):
            return "room"

    # style may include fill:rgba(...) or fill:none
    if "fill:none" in fill or "fill:transparent" in fill:
        # outline-only shapes often represent walls; prefer wall
        # but if group_label contains room -> choose room
        if "room" in g:
            return "room"
        return "wall"

    return "room"


def parse_svg(svg_path, image_size=(1024, 1024)):
    """
    Parse SVG and return (wall_mask, door_mask, room_mask) numpy uint8 arrays (H,W)
    """
    try:
        doc = minidom.parse(svg_path)
    except Exception as e:
        print(f"[ERR] parse failed for {svg_path}: {e}")
        return None, None, None

    svg_els = doc.getElementsByTagName("svg")
    vb_w = vb_h = None
    vb_x = vb_y = 0.0
    if svg_els and svg_els[0].hasAttribute("viewBox"):
        vb = svg_els[0].getAttribute("viewBox").strip().split()
        if len(vb) == 4:
            try:
                vb_x, vb_y, vb_w, vb_h = map(float, vb)
            except Exception:
                vb_w = vb_h = None

    h, w = image_size
    if vb_w and vb_h:
        sx = w / vb_w
        sy = h / vb_h

        def transform(p):
            return int(round((p[0] - vb_x) * sx)), int(round((p[1] - vb_y) * sy))
    else:
        def transform(p):
            return int(round(p[0])), int(round(p[1]))

    wall_mask = np.zeros((h, w), dtype=np.uint8)
    door_mask = np.zeros((h, w), dtype=np.uint8)
    room_mask = np.zeros((h, w), dtype=np.uint8)

    tags = ("polygon", "polyline", "rect", "path")
    for tag in tags:
        for el in doc.getElementsByTagName(tag):
            try:
                group_label = get_ancestor_group_label(el)
                style = el.getAttribute("style") or ""
                # try to extract fill color from style
                fill_color = ""
                if "fill:" in style:
                    try:
                        fill_color = style.split("fill:")[1].split(";")[0].strip()
                    except Exception:
                        fill_color = ""
                # classification
                typ = None

                # extract geometry
                pts = []
                if tag in ("polygon", "polyline"):
                    pts = points_from_str(el.getAttribute("points") or "")
                elif tag == "rect":
                    x = float(el.getAttribute("x") or 0.0)
                    y = float(el.getAttribute("y") or 0.0)
                    w_rect = float(el.getAttribute("width") or 0.0)
                    h_rect = float(el.getAttribute("height") or 0.0)
                    pts = [
                        (x, y),
                        (x + w_rect, y),
                        (x + w_rect, y + h_rect),
                        (x, y + h_rect),
                    ]
                elif tag == "path":
                    d = el.getAttribute("d") or ""
                    sampled = sample_path_d(d, n_samples=250)
                    pts = sampled

                if not pts:
                    # nothing to draw
                    continue

                # transform points to image pixel coords and convert to int32 array for cv2
                pts_px = []
                for p in pts:
                    try:
                        px, py = transform(p)
                        # clamp into image dims
                        if px < -1000 or py < -1000 or px > 10000 or py > 10000:
                            continue
                        pts_px.append([int(px), int(py)])
                    except Exception:
                        continue
                if len(pts_px) < 3 and tag != "rect":
                    # a valid polygon needs >=3 points; skip otherwise
                    continue

                # determine type using group label and fill_color
                typ = classify_element(el, fill_color, group_label)

                # fill into masks
                arr = np.array(pts_px, dtype=np.int32)
                if typ == "wall":
                    cv2.fillPoly(wall_mask, [arr], 255)
                elif typ == "door":
                    cv2.fillPoly(door_mask, [arr], 255)
                else:
                    cv2.fillPoly(room_mask, [arr], 255)
            except Exception:
                continue

    doc.unlink()
    return wall_mask, door_mask, room_mask


def create_overlay(base_img_path, wall_mask, door_mask, room_mask):
    if not os.path.exists(base_img_path):
        return None
    base = cv2.imread(base_img_path)
    if base is None:
        return None
    overlay = base.copy()
    # color priority: walls (dark) over doors? We'll overlay rooms, doors, walls.
    overlay[room_mask > 0] = (180, 255, 180)  # light green
    overlay[door_mask > 0] = (255, 120, 120)  # red-ish
    overlay[wall_mask > 0] = (60, 60, 60)  # dark gray
    blend = cv2.addWeighted(base, 0.5, overlay, 0.5, 0)
    return blend


def batch_parse(base_dir=INPUT_BASE, save_dir=OUTPUT_BASE):
    os.makedirs(save_dir, exist_ok=True)
    folders = sorted(os.listdir(base_dir))
    for f in folders:
        folder_path = os.path.join(base_dir, f)
        if not os.path.isdir(folder_path) or not f.isdigit():
            continue

        svg_path = os.path.join(folder_path, "model.svg")
        img_path = os.path.join(folder_path, "F1_original.png")

        if not os.path.exists(svg_path):
            print(f"[SKIP] {f}: no model.svg")
            continue

        # determine canvas size using base image if available
        img_size = (DEFAULT_CANV, DEFAULT_CANV)
        base_img = None
        if os.path.exists(img_path):
            base_img = cv2.imread(img_path)
            if base_img is not None:
                img_size = base_img.shape[:2]  # (h, w)
            else:
                print(
                    f"[WARN] {f}: failed to read {img_path}, using default size {DEFAULT_CANV}"
                )

        print(f"[PROCESS] {f} (size {img_size[1]}x{img_size[0]})")
        walls, doors, rooms = parse_svg(svg_path, image_size=img_size)

        out_dir = os.path.join(save_dir, f)
        os.makedirs(out_dir, exist_ok=True)

        # ensure masks are available; if any is None, create blank
        h, w = img_size
        if walls is None:
            walls = np.zeros((h, w), dtype=np.uint8)
        if doors is None:
            doors = np.zeros((h, w), dtype=np.uint8)
        if rooms is None:
            rooms = np.zeros((h, w), dtype=np.uint8)

        # write masks (as single-channel PNG)
        cv2.imwrite(os.path.join(out_dir, "walls.png"), walls)
        cv2.imwrite(os.path.join(out_dir, "doors.png"), doors)
        cv2.imwrite(os.path.join(out_dir, "rooms.png"), rooms)

        # overlay for quick visual check
        overlay = None
        if base_img is not None:
            overlay = create_overlay(img_path, walls, doors, rooms)
            if overlay is not None:
                cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)
                print(f"[OK] {f} → masks + overlay")
            else:
                print(f"[OK] {f} → masks (overlay creation failed)")
        else:
            print(f"[OK] {f} → masks (no base image for overlay)")

    print("[DONE]")


if __name__ == "__main__":
    batch_parse()
