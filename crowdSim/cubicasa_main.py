"""
cubicasa_main.py
================
Runs the crowd simulation on a CubiCasa5k SVG floor plan and renders it
with **matplotlib** (no Taichi GUI required).

Usage
-----
# Default floor plan
.crowdSim\\Scripts\\python.exe cubicasa_main.py

# Custom SVG
.crowdSim\\Scripts\\python.exe cubicasa_main.py --svg cubicasa5k/high_quality/10004/model.svg

# Headless: skip live window, save PNG snapshots every N frames
.crowdSim\\Scripts\\python.exe cubicasa_main.py --no-gui --snapshot-every 60
"""

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# CLI args (parse before importing anything heavy)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CubiCasa5k Crowd Simulation (matplotlib renderer)")
parser.add_argument(
    "--svg",
    default="cubicasa5k/high_quality/103/model.svg",
    help="Path to a CubiCasa5k model.svg file",
)
parser.add_argument("--seed",            type=int, default=42,  help="Random seed")
parser.add_argument("--no-gui",          action="store_true",   help="Headless mode (no window)")
parser.add_argument("--snapshot-every",  type=int, default=0,   help="Save a PNG snapshot every N frames (0=off)")
parser.add_argument("--max-frames",      type=int, default=3000,help="Stop after this many frames (headless)")
args = parser.parse_args()

svg_path = args.svg
if not os.path.isabs(svg_path):
    svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), svg_path)

if not os.path.exists(svg_path):
    print(f"ERROR: SVG not found: {svg_path}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Taichi init (must happen before importing sim)
# ---------------------------------------------------------------------------
import taichi as ti
ti.init(arch=ti.cpu, log_level=ti.WARN)   # CPU is safest for headless envs

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
crowdsim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crowdSim")
sys.path.insert(0, crowdsim_dir)

from constants import GRID_W, GRID_H, NUM_AGENTS, TIME_STEP
import sim
from metrics import MetricsManager
from recorder import SimulationRecorder
from svg_parser import parse_svg_floorplan, save_debug_png
from heatmap_viz import generate_heatmaps

# ---------------------------------------------------------------------------
# Parse and initialise
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  CubiCasa5k Crowd Simulation  (matplotlib renderer)")
print(f"  Floor plan : {svg_path}")
print(f"{'='*60}\n")

grid_py, doors_py, floor_mask = parse_svg_floorplan(svg_path, GRID_W, GRID_H)

svg_folder = os.path.basename(os.path.dirname(svg_path))
results_dir = os.path.join(crowdsim_dir, "results", f"cubicasa_{svg_folder}")
os.makedirs(results_dir, exist_ok=True)

debug_png_path = os.path.join(results_dir, "parsed_floorplan.png")
save_debug_png(grid_py, doors_py, floor_mask, out_path=debug_png_path)
print(f"[main] Floor-plan debug image → {debug_png_path}")

sim.initialize_from_floorplan(grid_py, doors_py, floor_mask=floor_mask, seed=args.seed)
sim.update_spatial_hash()

# ---------------------------------------------------------------------------
# Build a static background image: walls=dark, floor=light, outdoor=grey
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib
if args.no_gui:
    matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Convert grid / mask to a NumPy RGB canvas once
bg = np.zeros((GRID_H, GRID_W, 3), dtype=np.float32)
for x in range(GRID_W):
    for y in range(GRID_H):
        if grid_py[x][y] == 1:
            bg[y, x] = [0.18, 0.18, 0.20]       # dark wall
        elif floor_mask[x][y] == 1:
            bg[y, x] = [0.95, 0.94, 0.90]        # warm floor
        else:
            bg[y, x] = [0.55, 0.65, 0.55]        # outdoor / exterior

# Door positions
door_xs = [d[0] for d in doors_py]
door_ys = [d[1] for d in doors_py]

# ---------------------------------------------------------------------------
# Matplotlib figure setup
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")
ax.axis("off")

im_bg    = ax.imshow(bg, origin="lower", extent=[0, GRID_W, 0, GRID_H], zorder=0)
door_sc  = ax.scatter(door_xs, door_ys, c="#00e676", s=120, marker="s",
                      zorder=2, label="Exits", linewidths=0)
agent_sc = ax.scatter([], [], c=[], cmap="plasma", vmin=0, vmax=3.0,
                      s=18, zorder=3, linewidths=0, label="Agents")

ax.set_xlim(0, GRID_W)
ax.set_ylim(0, GRID_H)
ax.set_aspect("equal")

title_obj = ax.set_title(
    f"CubiCasa  –  {svg_folder}   |  Alive: {NUM_AGENTS}  Exited: 0",
    color="white", fontsize=11, pad=6
)
fig.tight_layout(pad=0.3)

if not args.no_gui:
    plt.ion()
    plt.show(block=False)

# ---------------------------------------------------------------------------
# Recorder & metrics
# ---------------------------------------------------------------------------
rec     = SimulationRecorder(base_path=results_dir)
metrics = MetricsManager()

# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
frame      = 0
start_time = time.time()
SUBSTEPS   = 3
dt         = TIME_STEP / SUBSTEPS

print("[main] Simulation running — close the window or press Ctrl-C to stop.\n")

snapshot_dir = os.path.join(results_dir, "snapshots")

try:
    while True:
        # ---- Exit conditions ----
        if not args.no_gui:
            if not plt.fignum_exists(fig.number):
                print("[main] Window closed.")
                break
        else:
            if frame >= args.max_frames:
                print(f"[main] Reached max_frames={args.max_frames}.")
                break

        current_time = time.time() - start_time

        # ---- Simulation sub-steps ----
        for _ in range(SUBSTEPS):
            sim.compute_density()
            sim.predict_step(dt)
            sim.check_reached_goal(current_time)
            sim.update_spatial_hash()
            sim.resolve_collisions(2)
            sim.commit_next()

        sim.compute_metrics()
        sim.update_detailed_metrics(TIME_STEP, current_time)

        # ---- Read back agent positions & speeds ----
        alive_mask = np.array([sim.alive[i] for i in range(NUM_AGENTS)], dtype=bool)
        alive_count = int(alive_mask.sum())
        exited      = int(sim.exited_count[None])
        avg_spd     = float(sim.avg_speed[None])

        pos_np  = sim.pos.to_numpy()          # shape (N, 2)
        vel_np  = sim.vel.to_numpy()          # shape (N, 2)
        speeds  = np.linalg.norm(vel_np, axis=1)

        alive_pos    = pos_np[alive_mask]
        alive_speeds = speeds[alive_mask]

        # ---- Matplotlib update ----
        if alive_count > 0:
            agent_sc.set_offsets(alive_pos)
            agent_sc.set_array(alive_speeds)
        else:
            agent_sc.set_offsets(np.empty((0, 2)))

        title_obj.set_text(
            f"CubiCasa  –  {svg_folder}   |  "
            f"t={current_time:.1f}s   Alive: {alive_count}   Exited: {exited}/{NUM_AGENTS}   "
            f"AvgSpeed: {avg_spd:.2f}"
        )

        if not args.no_gui:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)          # yield to GUI event loop

        # ---- Metrics ----
        min_dist_sum = float(sim.min_dist_sum_frame[None])
        avg_min_dist = min_dist_sum / max(1, alive_count)
        metrics.update(
            t=current_time,
            exited=exited,
            speed=avg_spd,
            max_rho=float(sim.max_density_val[None]),
            collisions=int(sim.collision_frame_count[None]),
            min_dist=avg_min_dist,
            overloaded_count=int(sim.overloaded_cells_frame[None]),
        )

        # ---- Recorder ----
        sim.compute_velocity_field()
        rec.record_density(sim.density_grid)
        rec.record_velocity(sim.velocity_grid)

        # ---- Optional snapshot PNG ----
        if args.snapshot_every > 0 and frame % args.snapshot_every == 0:
            os.makedirs(snapshot_dir, exist_ok=True)
            snap_path = os.path.join(snapshot_dir, f"frame_{frame:05d}.png")
            fig.savefig(snap_path, dpi=100, facecolor=fig.get_facecolor())
            print(f"  snapshot → {snap_path}")

        # ---- Console progress ----
        if frame % 60 == 0:
            print(f"  t={current_time:6.1f}s  alive={alive_count:3d}  "
                  f"exited={exited:3d}  speed={avg_spd:.2f}")

        frame += 1

        # All agents exited?
        if alive_count == 0:
            print(f"\n[main] All agents exited at t={current_time:.2f}s (frame {frame})")
            if not args.no_gui:
                plt.pause(2.0)
            break

except KeyboardInterrupt:
    print("\n[main] Interrupted by user.")

# ---------------------------------------------------------------------------
# Save final figure + generate report
# ---------------------------------------------------------------------------
final_fig_path = os.path.join(results_dir, "final_frame.png")
fig.savefig(final_fig_path, dpi=150, facecolor=fig.get_facecolor())
print(f"[main] Final frame saved → {final_fig_path}")

if frame > 0:
    metrics.set_agent_data(
        sim.exit_time.to_numpy(),
        sim.total_dist.to_numpy(),
        sim.straight_dist.to_numpy(),
    )
    dt_step_avg = (time.time() - start_time) / frame
    scores = metrics.calculate_scores(total_cells=GRID_W * GRID_H, dt_step_avg=dt_step_avg)

    sim.dump_sim_state_npz(os.path.join(rec.get_output_dir(), "sim_state.npz"))
    rec.finish()
    dashboard_path = os.path.join(rec.get_output_dir(), "analysis_dashboard.png")
    metrics.plot_dashboard(scores=scores, save_path=dashboard_path)

    print("\n[main] Generating Density and Flow heatmaps...")
    generate_heatmaps(rec.get_output_dir())

    print(f"\n[main] Done!  Results → {rec.get_output_dir()}")
    print(f"       Frames simulated : {frame}")
    print(f"       Agents exited    : {sim.exited_count[None]}/{NUM_AGENTS}")
else:
    print("[main] No frames were simulated.")
