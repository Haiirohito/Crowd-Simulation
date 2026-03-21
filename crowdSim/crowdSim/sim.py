import random
import taichi as ti
import numpy as np
from constants import *
from maze import (
    make_maze_borders_and_rooms,
    place_doors_on_outer_wall,
    compute_distance_field_bfs,
)

max_agents = NUM_AGENTS
pos = ti.Vector.field(2, float, max_agents) # Agent position
vel = ti.Vector.field(2, float, max_agents) # Agent velocity
next_pos = ti.Vector.field(2, float, max_agents) # Agent next (to be predicted) position
next_vel = ti.Vector.field(2, float, max_agents) # Agent next (to be predicted) velocity
alive = ti.field(ti.i32, max_agents) # alive and in the room
goal_idx = ti.field(ti.i32, max_agents) # Here doors

grid = ti.field(ti.i32, shape=(GRID_W, GRID_H))
dist = ti.field(ti.i32, shape=(GRID_W, GRID_H))

door_x = ti.field(ti.i32, MAX_DOORS)
door_y = ti.field(ti.i32, MAX_DOORS)
num_doors = ti.field(ti.i32, ())

# Spatial hashing fields
grid_head = ti.field(ti.i32, shape=(GRID_W, GRID_H))
particle_next = ti.field(ti.i32, shape=max_agents)


# Density field for pathfinding
density_grid = ti.field(ti.f32, shape=(GRID_W, GRID_H))

# Velocity field for flow visualization (accumulated velocity per cell)
velocity_grid = ti.Vector.field(2, ti.f32, shape=(GRID_W, GRID_H))


# Metrics
avg_speed = ti.field(ti.f32, shape=())
exited_count = ti.field(ti.i32, shape=())
max_density_val = ti.field(ti.f32, shape=())    

# Detailed Metrics Fields
total_dist = ti.field(ti.f32, shape=max_agents)
exit_time = ti.field(ti.f32, shape=max_agents)
collision_frame_count = ti.field(ti.i32, shape=())
min_dist_sum_frame = ti.field(ti.f32, shape=())

# Scoring Fields
straight_dist = ti.field(ti.f32, shape=max_agents)
overloaded_cells_frame = ti.field(ti.i32, shape=())


def initialize(seed=42):
    grid_py = make_maze_borders_and_rooms(seed=seed)
    doors_py = place_doors_on_outer_wall(grid_py)
    dist_py = compute_distance_field_bfs(grid_py, doors_py)

    for i in range(GRID_W):
        for j in range(GRID_H):
            grid[i, j] = grid_py[i][j]
            dist[i, j] = dist_py[i][j] if dist_py[i][j] < 1000000 else 999999

    num_doors[None] = len(doors_py)
    for i, (dx, dy) in enumerate(doors_py):
        door_x[i] = dx
        door_y[i] = dy

    spawn_positions = []
    for x in range(1, GRID_W - 1):
        for y in range(1, GRID_H - 1):
            if grid_py[x][y] == 0:
                if (
                    GRID_W // 4 < x < 3 * GRID_W // 4
                    and GRID_H // 4 < y < 3 * GRID_H // 4
                ):
                    spawn_positions.append((x + 0.5, y + 0.5))

    random.shuffle(spawn_positions)
    if len(spawn_positions) < NUM_AGENTS:
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
        total_dist[i] = 0.0
        exit_time[i] = 0.0
        min_d = 1e9
        chosen = 0
        for k in range(num_doors[None]):
            dx = door_x[k] + 0.5
            dy = door_y[k] + 0.5
            dd = (px - dx) ** 2 + (py - dy) ** 2
        if alive[i] == 0:
            continue
        p = pos[i]
        gx = int(p.x)
        gy = int(p.y)
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            density_grid[gx, gy] += 1.0
    
    # Find max density and count overloaded cells
    max_val = 0.0
    overloaded = 0
    for i in range(GRID_W):
        for j in range(GRID_H):
            val = density_grid[i, j]
            if val > max_val:
                max_val = val
            if val > 4.0: # Safe threshold
                overloaded += 1
    max_density_val[None] = max_val
    overloaded_cells_frame[None] = overloaded


def initialize_from_floorplan(grid_py, doors_py, floor_mask=None, seed=42):
    """
    Initialise the simulation from a pre-parsed SVG floor plan.

    Parameters
    ----------
    grid_py    : 2-D list [GRID_W][GRID_H], 0=walkable, 1=wall
    doors_py   : list of (gx, gy) grid coords for door/exit locations
    floor_mask : 2-D list [GRID_W][GRID_H], 1=indoor floor (spawn zone)
                 If None, all free cells are eligible for spawning.
    seed       : random seed for agent placement
    """
    from maze import compute_distance_field_bfs

    # Clamp doors to MAX_DOORS
    doors_py = doors_py[:MAX_DOORS]
    if not doors_py:
        doors_py = [(GRID_W - 2, GRID_H // 2)]

    # Compute BFS distance field toward doors
    dist_py = compute_distance_field_bfs(grid_py, doors_py)

    # Copy to Taichi fields
    for i in range(GRID_W):
        for j in range(GRID_H):
            grid[i, j] = grid_py[i][j]
            dist[i, j] = dist_py[i][j] if dist_py[i][j] < 1000000 else 999999

    # Door fields
    num_doors[None] = len(doors_py)
    for i, (dx, dy) in enumerate(doors_py):
        door_x[i] = dx
        door_y[i] = dy

    # ---- Agent spawn positions: prefer indoor floor cells ----
    random.seed(seed)
    spawn_positions = []

    if floor_mask is not None:
        # Only spawn inside the detected floor area
        for x in range(1, GRID_W - 1):
            for y in range(1, GRID_H - 1):
                if grid_py[x][y] == 0 and floor_mask[x][y] == 1:
                    spawn_positions.append((x + 0.5, y + 0.5))

    if len(spawn_positions) < NUM_AGENTS:
        # Fallback: all free cells
        spawn_positions = []
        for x in range(1, GRID_W - 1):
            for y in range(1, GRID_H - 1):
                if grid_py[x][y] == 0:
                    spawn_positions.append((x + 0.5, y + 0.5))

    if not spawn_positions:
        spawn_positions = [(GRID_W // 2 + 0.5, GRID_H // 2 + 0.5)]

    random.shuffle(spawn_positions)

    # Initialise agent Taichi fields
    exited_count[None] = 0
    collision_frame_count[None] = 0
    min_dist_sum_frame[None] = 0.0
    overloaded_cells_frame[None] = 0

    for i in range(NUM_AGENTS):
        px, py = spawn_positions[i % len(spawn_positions)]
        pos[i] = ti.Vector([px, py])
        next_pos[i] = ti.Vector([px, py])
        vel[i] = ti.Vector([0.0, 0.0])
        next_vel[i] = ti.Vector([0.0, 0.0])
        alive[i] = 1
        total_dist[i] = 0.0
        exit_time[i] = 0.0

        # Assign nearest door as initial goal
        min_d = 1e9
        chosen = 0
        for k in range(num_doors[None]):
            ddx = door_x[k] + 0.5
            ddy = door_y[k] + 0.5
            dd = (px - ddx) ** 2 + (py - ddy) ** 2
            if dd < min_d:
                min_d = dd
                chosen = k
        goal_idx[i] = chosen

    # Straight-line distances for scoring
    for i in range(NUM_AGENTS):
        if alive[i] == 1:
            px, py = pos[i].x, pos[i].y
            g = goal_idx[i]
            ddx = door_x[g] + 0.5
            ddy = door_y[g] + 0.5
            straight_dist[i] = ((px - ddx) ** 2 + (py - ddy) ** 2) ** 0.5

    print(f"[sim] Initialized from floor plan: {NUM_AGENTS} agents, "
          f"{len(doors_py)} doors, {len(spawn_positions)} spawn cells")


@ti.kernel
def compute_density():
    for i, j in density_grid:
        density_grid[i, j] = 0.0
    
    for i in range(max_agents):
        if alive[i] == 1:
            p = pos[i]
            gx = int(p.x)
            gy = int(p.y)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                ti.atomic_add(density_grid[gx, gy], 1.0)


@ti.kernel
def compute_metrics():
    total_speed = 0.0
    count = 0
    for i in range(max_agents):
        if alive[i] == 1:
            total_speed += vel[i].norm()
            count += 1
    if count > 0:
        avg_speed[None] = total_speed / count
    else:
        avg_speed[None] = 0.0


@ti.kernel
def compute_velocity_field():
    # 1. Clear grid
    for i, j in velocity_grid:
        velocity_grid[i, j] = ti.Vector([0.0, 0.0])
    
    # 2. Accumulate velocities and counts
    # We can reuse density_grid to store counts temporarily if strict sync isn't an issue,
    # but density_grid is used for pathfinding. 
    # Let's use a local temporary approach or just iterate agents.
    # Parallel scatter add is fine.
    
    for i in range(max_agents):
        if alive[i] == 1:
            p = pos[i]
            v = vel[i]
            gx = int(p.x)
            gy = int(p.y)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                ti.atomic_add(velocity_grid[gx, gy], v)

    # 3. Normalize by density (count)
    # computed in compute_density just before this usually.
    # Note: density_grid contains the count of agents per cell from `compute_density` kernel.
    for i, j in velocity_grid:
        count = density_grid[i, j]
        if count > 0:
            velocity_grid[i, j] /= count



@ti.kernel
def update_detailed_metrics(dt: ti.f32, current_time: ti.f32):
    collision_frame_count[None] = 0
    min_dist_sum_frame[None] = 0.0
    active_count = 0

    for i in range(max_agents):
        if alive[i] == 0:
            continue
        
        # Update travel distance
        total_dist[i] += vel[i].norm() * dt
        
        # Check collisions and min dist
        p = pos[i]
        gx = int(p.x)
        gy = int(p.y)
        min_d = 999.0
        
        # Neighbor search for metrics
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j and alive[j] != 0:
                            pj = pos[j]
                            d = (p - pj).norm()
                            if d < min_d:
                                min_d = d
                            if d < AGENT_RADIUS * 2.0:
                                ti.atomic_add(collision_frame_count[None], 1)
                        j = particle_next[j]
        
        if min_d < 999.0:
            ti.atomic_add(min_dist_sum_frame[None], min_d)
            active_count += 1


@ti.kernel
def predict_step(dt: ti.f32):
    for i in range(max_agents):
        if alive[i] == 0:
            # keep data consistent
            next_pos[i] = pos[i]
            next_vel[i] = ti.Vector([0.0, 0.0])
            continue

        p = pos[i]
        v = vel[i]

        gx = int(p.x)
        gy = int(p.y)
        best = ti.Vector([0.0, 0.0])
        best_score = 999999.0

        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    if grid[nx, ny] == 0:
                        # Pathfinding cost: distance to goal + penalty for high density
                        dval = float(dist[nx, ny]) + DENSITY_WEIGHT * density_grid[nx, ny]
                        if dval < best_score:
                            best_score = dval
                            best = ti.Vector([nx + 0.5, ny + 0.5])

        if best_score >= 999999.0:
            best_dist_euc = 1e9
            for dx in ti.static(range(-1, 2)):
                for dy in ti.static(range(-1, 2)):
                    nx = gx + dx
                    ny = gy + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                        if grid[nx, ny] == 0:
                            ncx = nx + 0.5
                            ncy = ny + 0.5
                            min_dd = 1e9
                            for k in range(num_doors[None]):
                                dxk = ncx - (door_x[k] + 0.5)
                                dyk = ncy - (door_y[k] + 0.5)
                                ddk = dxk * dxk + dyk * dyk
                                if ddk < min_dd:
                                    min_dd = ddk
                            if min_dd < best_dist_euc:
                                best_dist_euc = min_dd
                                best = ti.Vector([ncx, ncy])
            if best_dist_euc > 1e8:
                best = ti.Vector([gx + 0.5, gy + 0.5])
        to_goal = best - p
        dist_to_goal = to_goal.norm() + 1e-6
        desired_vel = (to_goal / dist_to_goal) * MAX_SPEED

        jitter = ti.Vector([0.0, 0.0])
        jitter.x += 0.05 * (i % 7)
        jitter.y += 0.05 * ((i + 3) % 5)
        desired_vel += jitter

        # Social Force (Repulsion)
        sep = ti.Vector([0.0, 0.0])
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j:
                            pj = pos[j]
                            diff = p - pj
                            d2 = diff.norm_sqr()
                            if d2 < SOCIAL_RADIUS * SOCIAL_RADIUS:
                                dist = ti.sqrt(d2)
                                force_mag = SOCIAL_WEIGHT * ti.exp(-dist / 0.5)
                                if dist > 1e-5:
                                    sep += (diff / dist) * force_mag
                        j = particle_next[j]

        wall_rep = ti.Vector([0.0, 0.0])
        for sx in ti.static(range(-1, 2)):
            for sy in ti.static(range(-1, 2)):
                nx = int(p.x) + sx
                ny = int(p.y) + sy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    if grid[nx, ny] == 1:
                        wc = ti.Vector([nx + 0.5, ny + 0.5])
                        diff = p - wc
                        dlen = diff.norm()
                        if dlen < 1.0:
                            wall_rep += (diff / (dlen + 1e-6)) * (1.0 - dlen)

        steer = (
            (desired_vel - v) * GOAL_WEIGHT + wall_rep * 3.0 + sep
        )

        v_new = v + steer * dt
        speed = v_new.norm()
        if speed > MAX_SPEED:
            v_new = v_new / speed * MAX_SPEED

        newp = p + v_new * dt

        ngx = int(newp.x)
        ngy = int(newp.y)
        if 0 <= ngx < GRID_W and 0 <= ngy < GRID_H and grid[ngx, ngy] == 1:
            newp = p

        next_pos[i] = newp
        next_vel[i] = v_new


@ti.kernel
def check_reached_goal(current_time: ti.f32):
    for i in range(max_agents):
        if alive[i] == 1:
            p = pos[i]
            g = goal_idx[i]
            dx = door_x[g] + 0.5
            dy = door_y[g] + 0.5
            if (p - ti.Vector([dx, dy])).norm() < 1.5:
                alive[i] = 0
                exit_time[i] = current_time
                ti.atomic_add(exited_count[None], 1)
                pos[i] = ti.Vector([-1000.0, -1000.0])
                next_pos[i] = ti.Vector([-1000.0, -1000.0])


@ti.kernel
def update_spatial_hash():
    for i, j in grid_head:
        grid_head[i, j] = -1

    # Serialize the list building to avoid needing atomic_exch
    ti.loop_config(serialize=True)
    for i in range(max_agents):
        if alive[i] == 0:
            continue

        # We use next_pos for collision detection
        p = next_pos[i]
        gx = int(p.x)
        gy = int(p.y)

        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            # Manual linked list insertion (safe because serialized)
            particle_next[i] = grid_head[gx, gy]
            grid_head[gx, gy] = i
        else:
            particle_next[i] = -1


@ti.kernel
def resolve_collisions(iters: ti.i32):
    for _ in range(iters):
        for i in range(max_agents):
            if alive[i] == 0:
                continue
            pi = next_pos[i]
            gx = int(pi.x)
            gy = int(pi.y)

            # Check 3x3 neighboring cells
            for dx in ti.static(range(-1, 2)):
                for dy in ti.static(range(-1, 2)):
                    nx = gx + dx
                    ny = gy + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                        # Traverse linked list in this cell
                        j = grid_head[nx, ny]
                        while j != -1:
                            if i < j: # Check each pair only once (i < j)
                                if alive[j] != 0:
                                    pj = next_pos[j]
                                    diff = pi - pj
                                    d_sq = diff.norm_sqr()
                                    min_dist = AGENT_RADIUS * 2.0
                                    if d_sq < min_dist * min_dist:
                                        d = ti.sqrt(d_sq) + 1e-9
                                        pen = min_dist - d
                                        n = diff / d
                                        # Soften the collision response to reduce jitter
                                        move = n * (pen * 0.2)
                                        next_pos[i] += move
                                        next_pos[j] -= move
                                        pi = next_pos[i] # Update local pi for subsequent checks
                            j = particle_next[j]

        for i in range(max_agents):
            if alive[i] == 0:
                continue
            p = next_pos[i]
            cx = int(p.x)
            cy = int(p.y)

            for wx in ti.static(range(-1, 2)):
                for wy in ti.static(range(-1, 2)):
                    nx = cx + wx
                    ny = cy + wy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                        if grid[nx, ny] == 1:
                            closest_x = p.x
                            if closest_x < nx:
                                closest_x = nx
                            elif closest_x > nx + 1.0:
                                closest_x = nx + 1.0
                            closest_y = p.y
                            if closest_y < ny:
                                closest_y = ny
                            elif closest_y > ny + 1.0:
                                closest_y = ny + 1.0

                            diffx = p.x - closest_x
                            diffy = p.y - closest_y
                            dist_sq = diffx * diffx + diffy * diffy
                            r = AGENT_RADIUS

                            if dist_sq < r * r:
                                dist_len = ti.sqrt(dist_sq) + 1e-9

                                push_dir = ti.Vector([0.0, 0.0])

                                if dist_len < 1e-6:
                                    tmp = ti.Vector([p.x - (nx + 0.5), p.y - (ny + 0.5)])
                                    nd = tmp.norm() + 1e-9
                                    push_dir = tmp / nd
                                else:
                                    push_dir = ti.Vector([diffx / dist_len, diffy / dist_len])

                                pen = r - dist_len
                                next_pos[i] = next_pos[i] + push_dir * pen



@ti.kernel
def commit_next():
    for i in range(max_agents):
        pos[i] = next_pos[i]
        vel[i] = next_vel[i]


def dump_sim_state_npz(path="sim_state.npz"):
    np_grid = np.zeros((GRID_W, GRID_H), dtype=np.int32)
    np_dist = np.zeros((GRID_W, GRID_H), dtype=np.int32)
    for x in range(GRID_W):
        for y in range(GRID_H):
            np_grid[x, y] = grid[x, y]
            np_dist[x, y] = dist[x, y]

    np_doors = []
    for i in range(num_doors[None]):
        np_doors.append([door_x[i], door_y[i]])
    np_doors = np.array(np_doors, dtype=np.int32)

    np.savez(path, grid=np_grid, dist=np_dist, doors=np_doors)
    print("Saved sim_state.npz")