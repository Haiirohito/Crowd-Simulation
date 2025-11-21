import os
import cv2
import numpy as np
from xml.dom import minidom


def parse_svg(svg_path, image_size=(512, 512)):
    """
    Parse CubiCasa5K SVG and create wall, door, and room masks.
    Handles grouped <polygon> and color variations.
    """
    doc = minidom.parse(svg_path)
    doc.getElementsByTagName("polygon")

    h, w = image_size
    wall_mask = np.zeros((h, w), np.uint8)
    door_mask = np.zeros((h, w), np.uint8)
    room_mask = np.zeros((h, w), np.uint8)

    svg = doc.getElementsByTagName("svg")[0]
    if svg.hasAttribute("viewBox"):
        viewbox = list(map(float, svg.getAttribute("viewBox").split()))
        vb_x, vb_y, vb_w, vb_h = viewbox
        sx = w / vb_w
        sy = h / vb_h
    else:
        sx = sy = 1.0

    current_group = ""

    for node in doc.getElementsByTagName("*"):
        if node.tagName == "g" and node.hasAttribute("id"):
            current_group = node.getAttribute("id").lower()

        elif node.tagName == "polygon" and node.hasAttribute("points"):
            style = node.getAttribute("style").lower()
            pts = []
            for pt in node.getAttribute("points").strip().split():
                try:
                    x, y = map(float, pt.split(","))
                    pts.append([int(x * sx), int(y * sy)])
                except:  # noqa: E722
                    continue
            if len(pts) == 0:
                continue
            pts = np.array(pts, np.int32)

            fill_color = ""
            if "fill:" in style:
                fill_color = style.split("fill:")[1].split(";")[0].strip()

            # Classification
            if (
                "door" in current_group
                or "door" in style
                or fill_color in ["#ff0000", "#ff7f00"]
            ):
                cv2.fillPoly(door_mask, [pts], 255)
            elif "wall" in current_group or fill_color in [
                "#000000",
                "#010101",
                "#1a1a1a",
            ]:
                cv2.fillPoly(wall_mask, [pts], 255)
            elif "room" in current_group or fill_color in [
                "#f5deb3",
                "#bebebe",
                "#e5e5e5",
                "#dcdcdc",
            ]:
                cv2.fillPoly(room_mask, [pts], 255)
            else:
                if fill_color not in ["none", "transparent"]:
                    cv2.fillPoly(room_mask, [pts], 255)

    doc.unlink()
    return wall_mask, door_mask, room_mask


def create_overlay(base_img_path, wall_mask, door_mask, room_mask):
    """
    Create and return a color overlay image for visual verification.
    """
    if not os.path.exists(base_img_path):
        return None

    base = cv2.imread(base_img_path)
    if base is None:
        return None

    overlay = base.copy()
    overlay[room_mask > 0] = (180, 255, 180)
    overlay[door_mask > 0] = (255, 120, 120)
    overlay[wall_mask > 0] = (60, 60, 60)

    result = cv2.addWeighted(base, 0.5, overlay, 0.5, 0)
    return result


def batch_parse(base_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    folders = sorted(os.listdir(base_dir))

    for f in folders:
        svg_path = os.path.join(base_dir, f, "model.svg")
        img_path = os.path.join(base_dir, f, "F1_original.png")  # use for overlay

        if not os.path.exists(svg_path):
            continue

        try:
            # --- read the base image for actual dimensions ---
            base_img = None
            if os.path.exists(img_path):
                base_img = cv2.imread(img_path)
                if base_img is None:
                    print(f"[WARN] Failed to read base image for {f}")
                    h = w = 1024
                else:
                    h, w = base_img.shape[:2]
            else:
                print(f"[WARN] Missing base image for {f}, using default size.")
                h = w = 1024

            # --- parse SVG with that size ---
            walls, doors, rooms = parse_svg(svg_path, image_size=(h, w))

            out_dir = os.path.join(save_dir, f)
            os.makedirs(out_dir, exist_ok=True)

            cv2.imwrite(os.path.join(out_dir, "walls.png"), walls)
            cv2.imwrite(os.path.join(out_dir, "doors.png"), doors)
            cv2.imwrite(os.path.join(out_dir, "rooms.png"), rooms)

            # --- overlay creation (only if we have a valid image) ---
            if base_img is not None:
                overlay = base_img.copy()
                overlay[rooms > 0] = (180, 255, 180)  # light green
                overlay[doors > 0] = (255, 120, 120)  # red
                overlay[walls > 0] = (60, 60, 60)  # dark gray
                blend = cv2.addWeighted(base_img, 0.5, overlay, 0.5, 0)

                overlay_path = os.path.join(out_dir, "overlay.png")
                ok = cv2.imwrite(overlay_path, blend)
                if ok:
                    print(f"[OK] Processed {f} ({w}x{h}) → overlay.png saved")
                else:
                    print(f"[WARN] Overlay not written for {f}")
            else:
                print(f"[OK] Processed {f} (masks only, no overlay)")

        except Exception as e:
            print(f"[FAIL] {f}: {e}")
