import taichi as ti

try:
    ti.init(arch=ti.gpu)
except Exception:
    try:
        ti.init(arch=ti.cuda)
    except Exception:
        ti.init(arch=ti.cpu)

from constants import *
import sim
import render
from metrics import MetricsManager
import time
import os
from recorder import SimulationRecorder

sim.initialize(seed=42)
sim.update_spatial_hash() # Prime the hash for the first frame
render.make_img()
render.cache_static_map()

rec = SimulationRecorder()
rec.save_maze_image(render.static_img)

metrics = MetricsManager()
gui = ti.GUI("Taichi Crowd Maze Escape - split", (WINDOW_SIZE, WINDOW_SIZE))
frame = 0
start_time = time.time()

while gui.running:
    # Substepping for stability
    SUBSTEPS = 3
    dt = TIME_STEP / SUBSTEPS
    current_time = time.time() - start_time
    
    for _ in range(SUBSTEPS):
        sim.compute_density()
        sim.predict_step(dt)
        sim.check_reached_goal(current_time)
        sim.update_spatial_hash()
        sim.resolve_collisions(2) # 2 iterations per substep is enough
        sim.commit_next()
    
    # Metrics Update
    sim.compute_metrics()
    sim.update_detailed_metrics(TIME_STEP, current_time)
    
    # Collect data
    alive_count = sum([int(sim.alive[i]) for i in range(NUM_AGENTS)])
    active_agents = NUM_AGENTS - sim.exited_count[None] # Approx
    
    # Calculate avg min dist from sum
    min_dist_sum = sim.min_dist_sum_frame[None]
    avg_min_dist = min_dist_sum / max(1, alive_count)
    
    metrics.update(
        t=current_time,
        exited=sim.exited_count[None],
        speed=sim.avg_speed[None],
        max_rho=sim.max_density_val[None],
        collisions=sim.collision_frame_count[None],
        min_dist=avg_min_dist,
        overloaded_count=sim.overloaded_cells_frame[None]
    )

    render.draw(gui)
    rec.capture_frame(render.img)

    avg_spd = sim.avg_speed[None]
    exited = sim.exited_count[None]
    
    gui.text(
        f"Alive: {alive_count}  Exited: {exited}",
        (0.01, 0.95),
        color=0x000000,
        font_size=20
    )
    gui.text(
        f"Avg Speed: {avg_spd:.2f} m/s",
        (0.01, 0.90),
        color=0x000000,
        font_size=20
    )

    gui.show()
    frame += 1

# End of simulation analysis
print("Simulation finished. Generating report...")
metrics.set_agent_data(
    sim.exit_time.to_numpy(),
    sim.total_dist.to_numpy(),
    sim.straight_dist.to_numpy()
)

dt_step_avg = (time.time() - start_time) / frame if frame > 0 else 0.0
scores = metrics.calculate_scores(total_cells=GRID_W * GRID_H, dt_step_avg=dt_step_avg)

rec.finish()
dashboard_path = os.path.join(rec.get_output_dir(), "analysis_dashboard.png")
metrics.plot_dashboard(scores=scores, save_path=dashboard_path)
