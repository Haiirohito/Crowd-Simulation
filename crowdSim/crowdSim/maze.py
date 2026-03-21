import random
from collections import deque
from constants import GRID_W, GRID_H, MAX_DOORS


def make_maze_borders_and_rooms(seed=42):
    W, H = GRID_W, GRID_H
    g = [[0 for _ in range(H)] for _ in range(W)]

    # outer walls
    for x in range(W):
        g[x][0] = 1
        g[x][H - 1] = 1
    for y in range(H):
        g[0][y] = 1
        g[W - 1][y] = 1

    def add_block(x1, y1, x2, y2):
        for x in range(x1, x2 + 1):
            g[x][y1] = 1
            g[x][y2] = 1
        for y in range(y1, y2 + 1):
            g[x1][y] = 1
            g[x2][y] = 1

    add_block(6, 6, W - 7, H - 7)
    add_block(12, 12, W - 13, H - 13)
    add_block(18, 18, W - 19, H - 19)

    fracs = [0.5, 0.33, 0.67]

    guaranteed_openings = []

    for f in fracs:
        guaranteed_openings.append((int(W * f), 1 + 1))
        guaranteed_openings.append((int(W * f), H - 2 - 1))
        guaranteed_openings.append((1 + 1, int(H * f)))
        guaranteed_openings.append((W - 2 - 1, int(H * f)))

    for f in fracs:
        guaranteed_openings.append((int(W * f), 6))
        guaranteed_openings.append((int(W * f), H - 7))
        guaranteed_openings.append((6, int(H * f)))
        guaranteed_openings.append((W - 7, int(H * f)))

    for f in fracs:
        guaranteed_openings.append((int(W * f), 12))
        guaranteed_openings.append((int(W * f), H - 13))
        guaranteed_openings.append((12, int(H * f)))
        guaranteed_openings.append((W - 13, int(H * f)))

    for f in fracs:
        guaranteed_openings.append((int(W * f), 18))
        guaranteed_openings.append((int(W * f), H - 19))
        guaranteed_openings.append((18, int(H * f)))
        guaranteed_openings.append((W - 19, int(H * f)))

    guaranteed_openings += [
        (W // 2, 6),
        (W // 2, H - 7),
        (6, H // 2),
        (W - 7, H // 2),
        (12, 12 + 5),
        (W - 13, H - 13 - 5),
        (18, 18 + 3),
        (W - 19 - 3, H - 19),
    ]

    for ox, oy in guaranteed_openings:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    g[nx][ny] = 0

    random.seed(seed)
    for i in range(80):
        bx = random.randint(3, W - 4)
        by = random.randint(3, H - 4)
        bw = random.randint(2, 6)
        bh = random.randint(2, 6)
        for x in range(max(1, bx), min(W - 1, bx + bw)):
            for y in range(max(1, by), min(H - 1, by + bh)):
                if random.random() < 0.9:
                    g[x][y] = 1

    for x in range(W // 3, 2 * W // 3):
        for y in range(H // 3, 2 * H // 3):
            if random.random() < 0.02:
                g[x][y] = 1
            else:
                g[x][y] = 0

    g = _ensure_connectivity(g, seed=seed)

    return g


def _ensure_connectivity(g, seed=42):
    W, H = GRID_W, GRID_H

    def bfs_from_sources(sources):
        INF = 10**9
        dist = [[INF for _ in range(H)] for _ in range(W)]
        q = deque()
        for sx, sy in sources:
            dist[sx][sy] = 0
            q.append((sx, sy))
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < W and 0 <= ny < H:
                    if g[nx][ny] == 0 and dist[nx][ny] > dist[x][y] + 1:
                        dist[nx][ny] = dist[x][y] + 1
                        q.append((nx, ny))
        return dist

    central_sources = []
    for x in range(W // 4 + 1, 3 * W // 4):
        for y in range(H // 4 + 1, 3 * H // 4):
            if g[x][y] == 0:
                central_sources.append((x, y))

    if not central_sources:
        cx, cy = W // 2, H // 2
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    g[nx][ny] = 0
        for x in range(W // 4 + 1, 3 * W // 4):
            for y in range(H // 4 + 1, 3 * H // 4):
                if g[x][y] == 0:
                    central_sources.append((x, y))

    outer_candidates = []
    for x in range(1, W - 1):
        if g[x][0] == 0:
            outer_candidates.append((x, 0))
        if g[x][H - 1] == 0:
            outer_candidates.append((x, H - 1))
    for y in range(1, H - 1):
        if g[0][y] == 0:
            outer_candidates.append((0, y))
        if g[W - 1][y] == 0:
            outer_candidates.append((W - 1, y))

    if not outer_candidates:
        g[W // 2][0] = 0
        g[W // 2][H - 1] = 0
        g[0][H // 2] = 0
        outer_candidates = [(W // 2, 0), (W // 2, H - 1), (0, H // 2)]

    dist = bfs_from_sources(central_sources)

    reachable_outer = []
    for ox, oy in outer_candidates:
        if dist[ox][oy] < 10**8:
            reachable_outer.append((ox, oy))

    if reachable_outer:
        return g

    random.seed(seed)
    cx = W // 2
    cy = H // 2
    central_sources.sort(key=lambda s: (s[0] - cx) ** 2 + (s[1] - cy) ** 2)
    src = central_sources[0] if central_sources else (cx, cy)

    best_outer = None
    best_d = 1e9
    for ox, oy in outer_candidates:
        dd = (ox - src[0]) ** 2 + (oy - src[1]) ** 2
        if dd < best_d:
            best_d = dd
            best_outer = (ox, oy)

    tx, ty = best_outer
    sx, sy = src

    x, y = sx, sy
    g[x][y] = 0
    step_dir_x = 1 if tx > x else -1
    while x != tx:
        x += step_dir_x
        for wx in range(-1, 2):
            for wy in range(-1, 2):
                nx = x + wx
                ny = y + wy
                if 0 <= nx < W and 0 <= ny < H:
                    g[nx][ny] = 0
    step_dir_y = 1 if ty > y else -1
    while y != ty:
        y += step_dir_y
        for wx in range(-1, 2):
            for wy in range(-1, 2):
                nx = x + wx
                ny = y + wy
                if 0 <= nx < W and 0 <= ny < H:
                    g[nx][ny] = 0

    # widen corridor area
    for ox in range(max(1, x - 3), min(W - 1, x + 4)):
        for oy in range(max(1, y - 3), min(H - 1, y + 4)):
            g[ox][oy] = 0

    return g


def place_doors_on_outer_wall(grid_py):
    W, H = GRID_W, GRID_H
    candidates = []
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
        grid_py[W // 2][0] = 0
        grid_py[W // 2][H - 1] = 0
        grid_py[0][H // 2] = 0
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

    chosen = []
    if candidates:
        step = max(1, len(candidates) // MAX_DOORS)
        for i in range(0, len(candidates), step):
            chosen.append(candidates[i])
            if len(chosen) >= MAX_DOORS:
                break
    if not chosen:
        chosen = [(W - 2, H // 2)]
    return chosen[:MAX_DOORS]


def compute_distance_field_bfs(grid_py, doors):
    W, H = GRID_W, GRID_H
    INF = 10**8
    d = [[INF for _ in range(H)] for _ in range(W)]
    q = deque()
    for dx, dy in doors:
        if grid_py[dx][dy] == 0:
            d[dx][dy] = 0
            q.append((dx, dy))
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < W and 0 <= ny < H:
                if grid_py[nx][ny] == 0 and d[nx][ny] > d[x][y] + 1:
                    d[nx][ny] = d[x][y] + 1
                    q.append((nx, ny))
    return d
