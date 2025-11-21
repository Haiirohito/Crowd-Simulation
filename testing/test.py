import cv2
import numpy as np
import json
from pathlib import Path
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


def process_floor_plan(image_path, display=True):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # --- 1. Threshold walls ---
    _, wall_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

    # --- 2. Skeletonize walls ---
    skeleton = skeletonize(wall_mask // 255).astype(np.uint8) * 255

    # --- 3. Find room boundaries (contours) ---
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rooms = [
        cv2.approxPolyDP(c, 3, True).reshape(-1, 2).tolist()
        for c in contours
        if cv2.contourArea(c) > 500
    ]

    # --- 4. Detect doors (gap-based heuristic) ---
    doors = []
    lines = cv2.HoughLinesP(
        skeleton, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=5
    )
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if np.hypot(x2 - x1, y2 - y1) < 50:  # short segment = possible door
                doors.append([(int(x1), int(y1)), (int(x2), int(y2))])

    # --- Visualization ---
    if display:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # draw walls (red)
        vis[skeleton > 0] = (0, 0, 255)

        # draw room contours (green)
        for r in rooms:
            cv2.polylines(vis, [np.array(r, np.int32)], True, (0, 255, 0), 2)

        # draw doors (blue)
        for p1, p2 in doors:
            cv2.line(vis, p1, p2, (255, 0, 0), 2)

        plt.figure(figsize=(8, 8))
        plt.title(f"Processed {Path(image_path).name}")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return {"image": str(image_path), "rooms": rooms, "doors": doors}


def process_dataset(root_dir, output_json="dataset.json", display=True):
    dataset = []
    for img_file in Path(root_dir).glob("*.png"):
        data = process_floor_plan(img_file, display=display)
        if data:
            dataset.append(data)

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    return dataset


# Run dataset processing (walls, doors, room boundaries)
root_dir = "dataset/floor_plan_arch/Y"  # update if path differs
output_file = "floorplan_dataset.json"

dataset = process_dataset(root_dir, output_json=output_file, display=True)

print(f"Processed {len(dataset)} floor plans")
print(f"Dataset saved to {output_file}")

# Preview first entry
if dataset:
    from pprint import pprint

    pprint(dataset[0])
