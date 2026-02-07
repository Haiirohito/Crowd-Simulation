import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob

from maze import (
    make_maze_borders_and_rooms,
    place_doors_on_outer_wall,
    compute_distance_field_bfs,
    GRID_W, GRID_H
)

def find_latest_results_dir(base_path="results"):
    dirs = [d for d in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(d)]
    if not dirs:
        return None
    # Sort by creation time (or name since it's timestamped)
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir

def generate_theoretical_flow(results_dir):
    print("Generating Theoretical Flow Map (Seed 42 reconstruction)...")
    
    grid_py = make_maze_borders_and_rooms(seed=42)
    doors_py = place_doors_on_outer_wall(grid_py)
    dist_py = compute_distance_field_bfs(grid_py, doors_py)
    
    W, H = GRID_W, GRID_H
    U = np.zeros((W, H))
    V = np.zeros((W, H))
    
    dist_np = np.array(dist_py)
    
    for x in range(1, W - 1):
        for y in range(1, H - 1):
            if grid_py[x][y] == 1:
                continue # Wall
            
            d_curr = dist_np[x, y]
            best_dx, best_dy = 0, 0
            min_n = d_curr
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x+dx, y+dy
                if dist_np[nx, ny] < min_n:
                    min_n = dist_np[nx, ny]
                    best_dx, best_dy = dx, dy
            
            U[x, y] = best_dx
            V[x, y] = best_dy

    plt.figure(figsize=(10, 10))
    
    X, Y = np.meshgrid(np.arange(H), np.arange(W))
    
    plt.imshow(np.array(grid_py).T, origin='lower', cmap='binary', alpha=0.3)
    
    
    x = np.arange(W)
    y = np.arange(H)
    Y_grid, X_grid = np.meshgrid(y, x)
    
    strm = plt.streamplot(x, y, U.T, V.T, color='blue', linewidth=1, density=2)
    
    plt.title('Theoretical Flow Map (Ideal Pathing)')
    plt.xlabel('X (Width)')
    plt.ylabel('Y (Height)')
    
    output_path = os.path.join(results_dir, "heatmap_theoretical_flow.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def generate_heatmaps(results_dir):
    data_path = os.path.join(results_dir, "density_data.npy")
    
    try:
        generate_theoretical_flow(results_dir)
    except Exception as e:
        print(f"Failed to generate theoretical flow: {e}")

    # 1. Density Heatmaps
    if os.path.exists(data_path):
        print(f"Loading density data from {data_path}...")
        density_data = np.load(data_path)
        
        # Peak
        peak_density_map = np.max(density_data, axis=0)
        plt.figure(figsize=(10, 10))
        plt.imshow(peak_density_map.T, origin='lower', cmap='inferno')
        plt.colorbar(label='Max Agents per Cell')
        plt.title('Peak Congestion Heatmap (Max Density per Cell)')
        output_path_peak = os.path.join(results_dir, "heatmap_peak.png")
        plt.savefig(output_path_peak)
        plt.close()
        print(f"Saved {output_path_peak}")

        # Average
        avg_density_map = np.mean(density_data, axis=0)
        plt.figure(figsize=(10, 10))
        plt.imshow(avg_density_map.T, origin='lower', cmap='viridis')
        plt.colorbar(label='Avg Agents per Cell')
        plt.title('Average Density Heatmap (Time-Averaged)')
        output_path_avg = os.path.join(results_dir, "heatmap_average.png")
        plt.savefig(output_path_avg)
        plt.close()
        print(f"Saved {output_path_avg}")
    else:
        print(f"No density_data.npy found in {results_dir} (Skipping density maps)")

    # 2. Actual Flow
    vel_path = os.path.join(results_dir, "velocity_data.npy")
    if os.path.exists(vel_path):
        print(f"Loading velocity data from {vel_path}...")
        try:
            vel_data = np.load(vel_path)

            avg_vel = np.mean(vel_data, axis=0)
            
            U = avg_vel[:, :, 0]
            V = avg_vel[:, :, 1]
            
            W, H = U.shape
            x = np.arange(W)
            y = np.arange(H)
            
            plt.figure(figsize=(10, 10))
            
            if os.path.exists(data_path):
                 avg_den = np.mean(np.load(data_path), axis=0)
                 plt.imshow(avg_den.T, origin='lower', cmap='Greys', alpha=0.5)
            
            
            speed = np.sqrt(U**2 + V**2)
            lw = 10 * speed / (speed.max() + 1e-6)
            
            plt.streamplot(x, y, U.T, V.T, color='purple', density=2, linewidth=lw.T)
            plt.title('Actual Flow Map (Time-Averaged Velocities)')
            output_path_vel = os.path.join(results_dir, "heatmap_actual_flow.png")
            plt.savefig(output_path_vel)
            plt.close()
            print(f"Saved {output_path_vel}")
            
        except Exception as e:
            print(f"Failed to process velocity data: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = find_latest_results_dir()

    if target_dir:
        generate_heatmaps(target_dir)
    else:
        print("No results directory found or provided.")
