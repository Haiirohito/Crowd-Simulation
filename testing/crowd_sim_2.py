import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import math
import random


# This function is correct and does not need changes.
def process_floor_plan(image_path, display_plot=True, line_thickness=3):
    """
    Extracts the wall centerline via skeletonization and finds door locations.
    Allows for increasing the centerline width using dilation.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, [], None, None, None

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])
    wall_mask = cv2.inRange(image, lower_black, upper_black)
    wall_mask_8u1 = cv2.threshold(wall_mask, 127, 255, cv2.THRESH_BINARY)[1]
    centerline_1px = cv2.ximgproc.thinning(wall_mask_8u1)

    final_centerline_img = centerline_1px
    if line_thickness > 1:
        dilation_kernel = np.ones((line_thickness, line_thickness), np.uint8)
        final_centerline_img = cv2.dilate(centerline_1px, dilation_kernel, iterations=1)

    # Create the grid matrix here, once.
    grid_matrix = np.where(final_centerline_img == 255, 0, 1).tolist()
    grid = Grid(matrix=grid_matrix)

    inverted_centerline = 255 - final_centerline_img
    visualization_img = cv2.cvtColor(inverted_centerline, cv2.COLOR_GRAY2BGR)

    lower_door_color = np.array([0, 0, 245])
    upper_door_color = np.array([0, 30, 255])
    door_mask = cv2.inRange(image, lower_door_color, upper_door_color)
    door_contours, _ = cv2.findContours(
        door_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    door_locations = []
    for contour in door_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            door_locations.append((cX, cY))
            cv2.circle(visualization_img, (cX, cY), 8, (0, 0, 255), -1)

    return grid, door_locations, visualization_img, image.shape


# --- REFACTORED SIMULATION FUNCTION ---
def find_and_draw_path(grid, doors, viz_img, shape, agent_id=1):
    """
    Finds a path for a single agent and draws it on an EXISTING visualization image.
    """
    height, width = shape[0], shape[1]

    # Find a random, walkable starting point
    start_point = None
    while True:
        random_x = random.randint(0, width - 1)
        random_y = random.randint(0, height - 1)
        test_node = grid.node(random_x, random_y)
        if test_node.walkable:
            start_point = (random_x, random_y)
            break

    print(f"Agent #{agent_id}: Found a valid random start at {start_point}.")
    start_node = grid.node(start_point[0], start_point[1])

    # Find the closest door
    closest_door = min(
        doors,
        key=lambda door: math.sqrt(
            (door[0] - start_point[0]) ** 2 + (door[1] - start_point[1]) ** 2
        ),
    )
    end_node = grid.node(closest_door[0], closest_door[1])

    # Find and draw the path
    finder = AStarFinder()
    path, runs = finder.find_path(start_node, end_node, grid)

    if not path:
        print(f"Agent #{agent_id}: No path could be found from {start_point}!")
    else:
        # Draw the path on the visualization image that was passed in
        for i in range(len(path) - 1):
            cv2.line(viz_img, path[i], path[i + 1], (255, 100, 0), 2)  # Blue-ish path
        cv2.circle(viz_img, start_point, 5, (0, 255, 0), -1)  # Green start point

    # This function now modifies viz_img directly, no need to return it


# --- CORRECTED EXECUTION BLOCK ---
if __name__ == "__main__":
    IMAGE_FILE = "floor_plan_256.png"  # Your image file
    NUMBER_OF_CROWS = 15

    # 1. Process the floor plan ONCE to get everything we need
    grid, doors, final_image, shape = process_floor_plan(IMAGE_FILE, display_plot=False)

    if final_image is not None and doors:
        # 2. Loop and draw paths for each agent on the SAME image
        for i in range(NUMBER_OF_CROWS):
            find_and_draw_path(grid, doors, final_image, shape, agent_id=i + 1)

        # 3. Display the single, final image with all the paths
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Paths for {NUMBER_OF_CROWS} Randomly Placed Agents")
        plt.axis("off")
        plt.show()
    elif not doors:
        print("Simulation failed: No doors were found in the image.")
    else:
        print("Simulation failed: Could not process the image file.")
