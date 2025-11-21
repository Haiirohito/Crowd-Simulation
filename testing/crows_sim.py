import cv2
import numpy as np
import pygame
import numba
import random
import collections


# ==============================================================================
#  1. DATA EXTRACTION FUNCTION
# ==============================================================================
def extract_centerline_and_doors(image_path, line_thickness=3):
    """
    Loads a floor plan image, extracts the wall centerline, door locations,
    the raw wall mask, and the room's outer contour.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, [], None, None

    # Create a mask of the thick walls
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])
    wall_mask = cv2.inRange(image, lower_black, upper_black)

    # Find the main room contour from the thick wall mask (for spawning)
    contours, _ = cv2.findContours(
        wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    room_contour = max(contours, key=cv2.contourArea) if contours else None

    # Skeletonize the mask to get a 1-pixel wide centerline for visualization
    centerline_1px = cv2.ximgproc.thinning(wall_mask)

    # Thicken the centerline for better visibility if needed
    final_centerline_img = centerline_1px
    if line_thickness > 1:
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        final_centerline_img = cv2.dilate(centerline_1px, kernel, iterations=1)

    # Create the visualization image (black lines on white background)
    visualization_img = cv2.cvtColor(255 - final_centerline_img, cv2.COLOR_GRAY2BGR)

    # Find door locations using their specific color
    lower_door = np.array([0, 0, 245])
    upper_door = np.array([0, 30, 255])
    door_mask = cv2.inRange(image, lower_door, upper_door)
    door_contours, _ = cv2.findContours(
        door_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    door_locations = []
    for c in door_contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            door_locations.append((cX, cY))

    return door_locations, visualization_img, wall_mask, room_contour


# ==============================================================================
#  2. POTENTIAL FIELD CALCULATION
# ==============================================================================
def create_distance_map(wall_map, doors):
    """
    Calculates a potential field where each walkable pixel's value is its
    distance to the nearest door. Uses a Breadth-First Search (BFS).
    """
    map_height, map_width = wall_map.shape
    distance_map = np.full((map_height, map_width), -1, dtype=np.int32)
    distance_map[wall_map == 1] = np.iinfo(np.int32).max

    queue = collections.deque()
    for x, y in doors:
        if 0 <= y < map_height and 0 <= x < map_width:
            queue.append((x, y, 0))  # (x, y, distance)
            distance_map[y, x] = 0

    while queue:
        x, y, dist = queue.popleft()

        # Check 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                if (
                    0 <= ny < map_height
                    and 0 <= nx < map_width
                    and distance_map[ny, nx] == -1
                ):
                    distance_map[ny, nx] = dist + 1
                    queue.append((nx, ny, dist + 1))

    distance_map[distance_map == -1] = np.iinfo(np.int32).max
    return distance_map


# ==============================================================================
#  3. NUMBA-ACCELERATED AGENT UPDATE LOGIC
# ==============================================================================
@numba.jit(nopython=True)
def update_agents_potential_field(agents, distance_map, speed, dt):
    """
    Updates agent positions by moving them "downhill" on the distance map.
    """
    num_agents = agents.shape[0]
    map_height, map_width = distance_map.shape

    for i in range(num_agents):
        if agents[i, 2] == 0:  # Skip inactive agents
            continue

        pos_x, pos_y = agents[i, 0], agents[i, 1]
        ix, iy = int(pos_x), int(pos_y)

        if distance_map[iy, ix] <= 1:  # Deactivate if at or next to a door
            agents[i, 2] = 0
            continue

        # Find the neighboring pixel with the lowest distance value
        min_dist = distance_map[iy, ix]
        best_dx, best_dy = 0, 0

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = ix + dx, iy + dy
                if 0 <= ny < map_height and 0 <= nx < map_width:
                    if distance_map[ny, nx] < min_dist:
                        min_dist = distance_map[ny, nx]
                        best_dx, best_dy = dx, dy

        # Move agent in the best direction
        if best_dx != 0 or best_dy != 0:
            norm = np.sqrt(float(best_dx * best_dx + best_dy * best_dy))
            vx = best_dx / norm * speed * dt
            vy = best_dy / norm * speed * dt
            agents[i, 0] += vx
            agents[i, 1] += vy

    return agents


# ==============================================================================
#  4. MAIN SIMULATION FUNCTION
# ==============================================================================


def run_simulation():
    print("🚀 Starting simulation setup...")

    # --- A. Environment Setup ---
    IMAGE_PATH = "dataset/2d floor plan/10017.png"
    print(f"   - Loading and processing: {IMAGE_PATH}...")

    doors, vis_img, wall_mask, room_contour = extract_centerline_and_doors(
        IMAGE_PATH, line_thickness=1
    )  # Using thin line for visualization
    if room_contour is None or not doors:
        print("   - ERROR: Could not load map or doors from image.")
        return

    MAP_HEIGHT, MAP_WIDTH = wall_mask.shape

    # ✅ --- THIS IS THE FIX ---
    # 1. Define agent size and create a thickened collision map
    AGENT_RADIUS = 3
    kernel_size = AGENT_RADIUS * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    collision_map = cv2.dilate(wall_mask, kernel, iterations=1)

    # 2. Create the distance map using the thickened collision_map
    print("   - Calculating potential field on buffered map...")
    distance_map = create_distance_map(collision_map, doors)
    # --- END FIX ---

    # --- B. Agent Setup ---
    NUM_AGENTS = 150
    AGENT_SPEED = 50.0
    print(f"   - Finding valid spawn points for {NUM_AGENTS} agents...")

    interior_mask = np.zeros_like(wall_mask)
    cv2.drawContours(interior_mask, [room_contour], -1, 255, -1)

    # Erode the interior to create a spawn buffer from the *original* walls
    spawn_kernel = np.ones((AGENT_RADIUS + 2, AGENT_RADIUS + 2), np.uint8)
    spawnable_mask = cv2.erode(interior_mask, spawn_kernel, iterations=1)

    valid_indices = np.argwhere(spawnable_mask > 0)

    if len(valid_indices) < NUM_AGENTS:
        print(
            f"   - WARNING: Not enough valid spawn points ({len(valid_indices)}). Reducing agent count."
        )
        NUM_AGENTS = len(valid_indices)

    agents = np.zeros((NUM_AGENTS, 3), dtype=np.float32)
    spawn_points = valid_indices[:, ::-1].tolist()
    chosen_indices = np.random.choice(len(spawn_points), NUM_AGENTS, replace=False)
    for i, idx in enumerate(chosen_indices):
        x, y = spawn_points[idx]
        agents[i, 0], agents[i, 1], agents[i, 2] = x, y, 1

    # --- C. Pygame Initialization & Main Loop ---
    pygame.init()
    screen = pygame.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
    pygame.display.set_caption("Crowd Simulation")
    clock = pygame.time.Clock()

    wall_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)
    wall_surface.blit(
        pygame.surfarray.make_surface(
            cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        ),
        (0, 0),
    )
    for x, y in doors:
        pygame.draw.circle(wall_surface, (0, 255, 0), (x, y), 6, 3)

    print("✅ Setup complete. Launching simulation window...")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = clock.tick(60) / 1000.0
        agents = update_agents_potential_field(agents, distance_map, AGENT_SPEED, dt)

        screen.fill((255, 255, 255))
        screen.blit(wall_surface, (0, 0))

        active_agents = agents[agents[:, 2] == 1]
        for i in range(active_agents.shape[0]):
            pos = (int(active_agents[i, 0]), int(active_agents[i, 1]))
            pygame.draw.circle(screen, (255, 0, 0), pos, AGENT_RADIUS)

        pygame.display.flip()

    pygame.quit()


# ==============================================================================
#  5. SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    run_simulation()
