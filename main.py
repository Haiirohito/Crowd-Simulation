# crowd_taichi.py
# Taichi crowd simulation with maze, doors, distance-field pathfinding, and collision avoidance
# Tested with Taichi 1.x APIs (should work with modern Taichi releases)

import math
import random
from collections import deque

import taichi as ti

ti.init(arch=ti.gpu)  # try GPU; falls back to CPU if not available

# ----------------------
# CONFIG
# ----------------------
WINDOW_SIZE = 800
GRID_W = 80  # grid width for the maze (cells)
GRID_H = 80  # grid height for the maze (cells)
CELL_PIX = WINDOW_SIZE // GRID_W
NUM_AGENTS = 250
AGENT_RADIUS = 0.18  # in cell units
MAX_SPEED = 3.0  # cells / second
SEPARATION_WEIGHT = 6.0
GOAL_WEIGHT = 5.0
TIME_STEP = 1.0 / 60.0

# ----------------------
# TAICHI FIELDS
# ----------------------
# agent state: position (x,y), velocity (x,y), target door index
max_agents = NUM_AGENTS
pos = ti.Vector.field(2, float, max_agents)
vel = ti.Vector.field(2, float, max_agents)
alive = ti.field(ti.i32, max_agents)  # 1 if active
goal_idx = ti.field(ti.i32, max_agents)

# grid occupancy: 0 = free, 1 = wall
grid = ti.field(ti.i32, shape=(GRID_W, GRID_H))

# distance field (integer distances): large = unreachable
dist = ti.field(ti.i32, shape=(GRID_W, GRID_H))

# door positions (list, small)
MAX_DOORS = 8
door_x = ti.field(ti.i32, MAX_DOORS)
door_y = ti.field(ti.i32, MAX_DOORS)
num_doors = ti.field(ti.i32, ())


# ----------------------
# HELPER FUNCTIONS (Python side)
# ----------------------
def make_maze_borders_and_rooms():
    """Create a simple box-like maze with corridors and openings
    Returns a 2D list of 0/1 for free/wall.
    """
    W, H = GRID_W, GRID_H
    g = [[0 for _ in range(H)] for _ in range(W)]
    # outer walls
    for x in range(W):
        g[x][0] = 1
        g[x][H - 1] = 1
    for y in range(H):
        g[0][y] = 1
        g[W - 1][y] = 1

    # add rectangular blocks to form a maze-like layout
    def add_block(x1, y1, x2, y2):
        for x in range(x1, x2 + 1):
            g[x][y1] = 1
            g[x][y2] = 1
        for y in range(y1, y2 + 1):
            g[x1][y] = 1
            g[x2][y] = 1

    # Create a few nested rooms with openings
    add_block(6, 6, W - 7, H - 7)
    add_block(12, 12, W - 13, H - 13)
    add_block(18, 18, W - 19, H - 19)

    # carve some corridors/openings in walls (door-like openings in interior walls)
    openings = [
        (W // 2, 6),
        (W // 2, H - 7),  # openings in second outer ring
        (6, H // 2),
        (W - 7, H // 2),  # side openings
        (12, 12 + 5),
        (W - 13, H - 13 - 5),
    ]
    for ox, oy in openings:
        if 0 <= ox < W and 0 <= oy < H:
            g[ox][oy] = 0

    # add random obstacles/maze passages inside
    random.seed(42)
    for i in range(80):
        bx = random.randint(3, W - 4)
        by = random.randint(3, H - 4)
        bw = random.randint(2, 6)
        bh = random.randint(2, 6)
        for x in range(max(1, bx), min(W - 1, bx + bw)):
            for y in range(max(1, by), min(H - 1, by + bh)):
                if random.random() < 0.9:
                    g[x][y] = 1

    # carve out a larger central room
    for x in range(W // 3, 2 * W // 3):
        for y in range(H // 3, 2 * H // 3):
            if random.random() < 0.02:
                g[x][y] = 1
            else:
                g[x][y] = 0

    return g


def place_doors_on_outer_wall(grid_py):
    """
    Place doors (exits) along the outer walls at free cells.
    Returns list of (x,y).
    """
    W, H = GRID_W, GRID_H
    candidates = []
    # pick some free cells along the outer walls
    for x in range(1, W - 1):
        if grid_py[x][0] == 0:
            candidates.append((x, 0))
        if grid_py[x][H - 1] == 0:
            candidates.append((x, H - 1))
    for y in range(1, H - 1):
        if grid_py[0][y] == 0:
            candidates.append((0, y))
        if grid_py[W - 1][y] == 0:
            candidates.append((W - 1, y))
    if not candidates:
        return [(W - 2, H // 2)]
    # choose up to MAX_DOORS spaced out
    chosen = []
    step = max(1, len(candidates) // MAX_DOORS)
    for i in range(0, len(candidates), step):
        chosen.append(candidates[i])
        if len(chosen) >= MAX_DOORS:
            break
    return chosen[:MAX_DOORS]


def compute_distance_field_bfs(grid_py, doors):
    """Compute distance (in Manhattan grid steps) from each free cell to nearest door using BFS."""
    W, H = GRID_W, GRID_H
    INF = 10**8
    d = [[INF for _ in range(H)] for _ in range(W)]
    q = deque()
    # initialize with doors
    for dx, dy in doors:
        if grid_py[dx][dy] == 0:
            d[dx][dy] = 0
            q.append((dx, dy))
    # BFS 4-neighbor
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < W and 0 <= ny < H:
                if grid_py[nx][ny] == 0 and d[nx][ny] > d[x][y] + 1:
                    d[nx][ny] = d[x][y] + 1
                    q.append((nx, ny))
    return d


# ----------------------
# INITIALIZE GRID, DOORS, DISTANCE FIELD, AGENTS
# ----------------------
grid_py = make_maze_borders_and_rooms()
doors_py = place_doors_on_outer_wall(grid_py)
dist_py = compute_distance_field_bfs(grid_py, doors_py)

# copy to taichi fields
for i in range(GRID_W):
    for j in range(GRID_H):
        grid[i, j] = grid_py[i][j]
        dist[i, j] = dist_py[i][j] if dist_py[i][j] < 1000000 else 999999

num_doors[None] = len(doors_py)
for i, (dx, dy) in enumerate(doors_py):
    door_x[i] = dx
    door_y[i] = dy

# spawn agents inside central area (free cells)
spawn_positions = []
for x in range(1, GRID_W - 1):
    for y in range(1, GRID_H - 1):
        if grid_py[x][y] == 0:
            # spawn in central-ish area
            if GRID_W // 4 < x < 3 * GRID_W // 4 and GRID_H // 4 < y < 3 * GRID_H // 4:
                spawn_positions.append((x + 0.5, y + 0.5))

random.shuffle(spawn_positions)
if len(spawn_positions) < NUM_AGENTS:
    # allow broader spawn
    spawn_positions = []
    for x in range(1, GRID_W - 1):
        for y in range(1, GRID_H - 1):
            if grid_py[x][y] == 0:
                spawn_positions.append((x + 0.5, y + 0.5))

for i in range(NUM_AGENTS):
    px, py = spawn_positions[i % len(spawn_positions)]
    pos[i] = ti.Vector([px, py])
    vel[i] = ti.Vector([0.0, 0.0])
    alive[i] = 1
    # choose nearest door initially (index)
    min_d = 1e9
    chosen = 0
    for k in range(num_doors[None]):
        dx = door_x[k] + 0.5
        dy = door_y[k] + 0.5
        dd = (px - dx) ** 2 + (py - dy) ** 2
        if dd < min_d:
            min_d = dd
            chosen = k
    goal_idx[i] = chosen


# ----------------------
# TAICHI KERNELS
# ----------------------
@ti.func
def sample_grid_cell(p):
    """Return integer grid cell coords clamped and grid value"""
    gx = int(p.x)
    gy = int(p.y)
    if gx < 0:
        gx = 0
    if gx >= GRID_W:
        gx = GRID_W - 1
    if gy < 0:
        gy = 0
    if gy >= GRID_H:
        gy = GRID_H - 1
    return gx, gy


@ti.kernel
def step_sim(dt: ti.f32):
    # move agents
    for i in range(max_agents):
        if alive[i] == 0:
            continue

        p = pos[i]
        v = vel[i]

        # sample my grid cell and its 4-neighbor to find a direction toward decreasing dist
        gx = int(p.x)
        gy = int(p.y)
        best = ti.Vector([0.0, 0.0])
        best_score = 1e9
        # check neighbors (including staying in same cell)
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    if grid[nx, ny] == 0:
                        # Prefer neighbor with lower distance
                        dval = dist[nx, ny]
                        if dval < best_score:
                            best_score = dval
                            # target point is center of cell
                            best = ti.Vector([nx + 0.5, ny + 0.5])

        # compute goal-seeking force toward center of chosen neighbor cell
        to_goal = best - p
        dist_to_goal = to_goal.norm() + 1e-6
        desired_vel = (to_goal / dist_to_goal) * MAX_SPEED

        # separation / collision avoidance: simple repulsive around radius
        sep = ti.Vector([0.0, 0.0])
        for j in range(max_agents):
            if j == i:
                continue
            if alive[j] == 0:
                continue
            pj = pos[j]
            diff = p - pj
            dlen = diff.norm()
            if dlen < 1e-6:
                continue
            if dlen < AGENT_RADIUS * 2.4:  # influence zone
                # stronger repulsion when very close
                sep += (diff / dlen) * (AGENT_RADIUS * 2.4 - dlen) / (dlen + 1e-6)

        # wall avoidance: if projected next position would be inside wall cell, add repulsion
        # sample a few rays
        wall_rep = ti.Vector([0.0, 0.0])
        for sx in ti.static(range(-1, 2)):
            for sy in ti.static(range(-1, 2)):
                nx = int(p.x) + sx
                ny = int(p.y) + sy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    if grid[nx, ny] == 1:
                        # vector from wall cell center to agent
                        wc = ti.Vector([nx + 0.5, ny + 0.5])
                        diff = p - wc
                        dlen = diff.norm()
                        if dlen < 1.0:
                            wall_rep += (diff / (dlen + 1e-6)) * (1.0 - dlen)

        # combine forces
        steer = (
            (desired_vel - v) * GOAL_WEIGHT + sep * SEPARATION_WEIGHT + wall_rep * 3.0
        )

        # integrate velocity
        v = v + steer * dt
        speed = v.norm()
        if speed > MAX_SPEED:
            v = v / speed * MAX_SPEED

        # integrate position
        newp = p + v * dt

        # clamp to stay outside walls: if newp would be inside wall cell, push back along gradient
        ngx = int(newp.x)
        ngy = int(newp.y)
        if 0 <= ngx < GRID_W and 0 <= ngy < GRID_H and grid[ngx, ngy] == 1:
            # push back to previous position and reduce velocity
            v = v * 0.0
            newp = p

        # write back
        pos[i] = newp
        vel[i] = v

        # Check if reached a door cell (exit)
        # Consider agent reached if its cell is a door coordinate
        gx2 = int(newp.x)
        gy2 = int(newp.y)
        for k in range(num_doors[None]):
            if gx2 == door_x[k] and gy2 == door_y[k]:
                # remove agent (escaped)
                alive[i] = 0
                vel[i] = ti.Vector([0.0, 0.0])
                break


# ----------------------
# RENDERING
# ----------------------
gui = ti.GUI("Taichi Crowd Maze Escape", (WINDOW_SIZE, WINDOW_SIZE))
frame = 0


def draw():
    # draw background
    img = ti.Vector.field(3, float, shape=(WINDOW_SIZE, WINDOW_SIZE))

    @ti.kernel
    def rasterize():
        for i, j in img:
            img[i, j] = ti.Vector([1.0, 1.0, 1.0])  # white background

        # walls
        for gx, gy in ti.ndrange(GRID_W, GRID_H):
            if grid[gx, gy] == 1:
                # draw cell as dark block
                x0 = int(gx * CELL_PIX)
                y0 = int(gy * CELL_PIX)
                for dx, dy in ti.ndrange(CELL_PIX, CELL_PIX):
                    ii = x0 + dx
                    jj = y0 + dy
                    if 0 <= ii < WINDOW_SIZE and 0 <= jj < WINDOW_SIZE:
                        img[ii, jj] = ti.Vector([0.2, 0.2, 0.2])

        # doors - draw as green openings
        for k in range(num_doors[None]):
            gx = door_x[k]
            gy = door_y[k]
            cx = int((gx + 0.5) * CELL_PIX)
            cy = int((gy + 0.5) * CELL_PIX)
            r = CELL_PIX // 2
            for ii, jj in ti.ndrange(WINDOW_SIZE, WINDOW_SIZE):
                # quick bounding box
                if abs(ii - cx) <= r and abs(jj - cy) <= r:
                    # draw small square
                    img[ii, jj] = ti.Vector([0.2, 0.9, 0.2])

        # agents
        for i in range(max_agents):
            if alive[i] == 0:
                continue
            p = pos[i]
            px = int(p.x * CELL_PIX)
            py = int(p.y * CELL_PIX)
            rpx = max(1, int(AGENT_RADIUS * CELL_PIX))
            for ii in range(-rpx, rpx + 1):
                for jj in range(-rpx, rpx + 1):
                    ii2 = px + ii
                    jj2 = py + jj
                    if 0 <= ii2 < WINDOW_SIZE and 0 <= jj2 < WINDOW_SIZE:
                        if ii * ii + jj * jj <= rpx * rpx:
                            img[ii2, jj2] = ti.Vector([0.1, 0.4, 1.0])

    rasterize()
    gui.set_image(img.to_numpy())


# ----------------------
# MAIN LOOP
# ----------------------
while gui.running:
    for _ in range(1):  # sub-steps if needed
        step_sim(TIME_STEP)

    draw()

    # overlay stats
    alive_count = sum([int(alive[i]) for i in range(NUM_AGENTS)])
    gui.text(
        f"Alive: {alive_count}  Agents: {NUM_AGENTS}  Doors: {num_doors[None]}",
        (0.01, 0.01),
        color=0x000000,
    )
    gui.show()
    frame += 1
