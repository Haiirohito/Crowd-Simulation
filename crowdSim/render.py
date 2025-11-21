import taichi as ti
from constants import *
import sim

static_img = None

def make_img():
    global img, static_img
    img = ti.Vector.field(3, float, shape=(WINDOW_SIZE, WINDOW_SIZE))
    static_img = ti.Vector.field(3, float, shape=(WINDOW_SIZE, WINDOW_SIZE))

@ti.kernel
def cache_static_map():
    # Clear background
    for i, j in static_img:
        static_img[i, j] = ti.Vector([1.0, 1.0, 1.0])

    # Draw Walls
    for gx, gy in ti.ndrange(GRID_W, GRID_H):
        if sim.grid[gx, gy] == 1:
            x0 = int(gx * CELL_PIX)
            y0 = int(gy * CELL_PIX)
            for dx, dy in ti.ndrange(CELL_PIX, CELL_PIX):
                ii = x0 + dx
                jj = y0 + dy
                if 0 <= ii < WINDOW_SIZE and 0 <= jj < WINDOW_SIZE:
                    static_img[ii, jj] = ti.Vector([0.2, 0.2, 0.2])

    # Draw Doors
    for k in range(sim.num_doors[None]):
        gx = sim.door_x[k]
        gy = sim.door_y[k]
        cx = int((gx + 0.5) * CELL_PIX)
        cy = int((gy + 0.5) * CELL_PIX)
        r = CELL_PIX // 2
        x0 = max(0, cx - r)
        x1 = min(WINDOW_SIZE, cx + r)
        y0 = max(0, cy - r)
        y1 = min(WINDOW_SIZE, cy + r)
        for ii in range(x0, x1):
            for jj in range(y0, y1):
                static_img[ii, jj] = ti.Vector([0.2, 0.9, 0.2])

@ti.kernel
def rasterize():
    # Copy static map to current image
    for i, j in img:
        img[i, j] = static_img[i, j]

    # Draw Agents
    for i in range(sim.max_agents):
        if sim.alive[i] == 0:
            continue
        p = sim.pos[i]
        px = int(p.x * CELL_PIX)
        py = int(p.y * CELL_PIX)
        rpx = max(1, int(AGENT_RADIUS * CELL_PIX))

        # Data-Driven Color
        color = ti.Vector([0.1, 0.4, 1.0]) # Default Blue

        # Mode 1: Speed (Blue -> Red)
        speed = sim.vel[i].norm()
        t_speed = ti.min(speed / MAX_SPEED, 1.0)
        color = ti.Vector([t_speed, 0.1, 1.0 - t_speed])

        # Mode 2: Density (Green -> Red) - DISABLED
        # gx = int(p.x)
        # gy = int(p.y)
        # if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
        #     dens = sim.density_grid[gx, gy]
        #     t_dens = ti.min(dens / 4.0, 1.0)
        #     pass

        for ii in range(-rpx, rpx + 1):
            for jj in range(-rpx, rpx + 1):
                ii2 = px + ii
                jj2 = py + jj
                if 0 <= ii2 < WINDOW_SIZE and 0 <= jj2 < WINDOW_SIZE:
                    if ii * ii + jj * jj <= rpx * rpx:
                        img[ii2, jj2] = color


def draw(gui):
    rasterize()
    gui.set_image(img) # Pass field directly to avoid GPU->CPU copy
