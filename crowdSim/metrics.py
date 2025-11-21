import matplotlib.pyplot as plt
import numpy as np
import time

class MetricsManager:
    def __init__(self):
        # Time series data
        self.times = []
        self.exited_counts = []
        self.avg_speeds = []
        self.max_densities = []
        self.collision_counts = []
        self.overloaded_counts = []
        self.avg_min_dists = []
        
        # Agent-specific data (populated at end or during)
        self.agent_exit_times = []
        self.agent_travel_dists = []
        
        self.start_time = time.time()
        self.sim_time = 0.0

    def update(self, t, exited, speed, max_rho, collisions, min_dist, overloaded_count):
        self.times.append(t)
        self.exited_counts.append(exited)
        self.avg_speeds.append(speed)
        self.max_densities.append(max_rho)
        self.collision_counts.append(collisions)
        self.avg_min_dists.append(min_dist)
        self.overloaded_counts.append(overloaded_count)
        self.sim_time = t

    def set_agent_data(self, exit_times, travel_dists, straight_dists):
        self.agent_exit_times = exit_times
        self.agent_travel_dists = travel_dists
        self.agent_straight_dists = straight_dists

    def calculate_scores(self, total_cells, dt_step_avg):
        # -------------------------------
        # 1. Extract Raw Metrics
        # -------------------------------
        T_total = self.times[-1] if self.times else 0

        total_agents = self.exited_counts[-1] if self.exited_counts else 0
        
        # t_90 computation
        if total_agents > 0:
            t_90_idx = next((i for i, x in enumerate(self.exited_counts)
                            if x >= 0.9 * total_agents), -1)
            t_90 = self.times[t_90_idx] if t_90_idx != -1 else T_total
        else:
            t_90 = 0

        valid_times = [t for t in self.agent_exit_times if t > 0]
        T_mean = np.mean(valid_times) if valid_times else 0
        T_95 = np.percentile(valid_times, 95) if valid_times else 0
        
        rho_max = np.max(self.max_densities) if self.max_densities else 0
        
        # rho_over = (space × time overload fraction)
        total_frames = len(self.times)
        total_space_time = total_cells * total_frames
        rho_over = (sum(self.overloaded_counts) / total_space_time) if total_space_time > 0 else 0
        
        d_min_avg = np.mean(self.avg_min_dists) if self.avg_min_dists else 0
        N_collisions = sum(self.collision_counts)

        # Detour index
        detour_indices = []
        for i in range(len(self.agent_travel_dists)):
            if self.agent_travel_dists[i] > 0 and self.agent_straight_dists[i] > 0:
                detour_indices.append(self.agent_travel_dists[i] /
                                    self.agent_straight_dists[i])
        D_mean = np.mean(detour_indices) if detour_indices else 1.0

        dt_step_ms = dt_step_avg * 1000.0  # Convert to ms


        # -------------------------------
        # 2. Correct Clamp Function
        # -------------------------------
        def clamp01(x):
            return max(0.0, min(1.0, x))


        # -------------------------------
        # 3. Tuned, Realistic Normalization
        # -------------------------------

        # Efficiency (lower is better)
        score_T_total = clamp01((300 - T_total) / 180)    # good=120s, bad=300s
        score_t90     = clamp01((200 - t_90) / 150)       # good=50s,  bad=200s

        # Safety (lower is better)
        score_rho_max  = clamp01((6.0 - rho_max) / 4.0)   # good=2, bad=6
        score_rho_over = clamp01((0.30 - rho_over) / 0.30)
        score_collisions = clamp01((800000 - N_collisions) / 700000)

        # Comfort
        score_d_min = clamp01((d_min_avg - 0.6) / 0.6)    # good=1.2m, bad=0.6m
        score_detour = clamp01((3.2 - D_mean) / 2.2)      # good=1.0, bad=3.2

        # Performance (auto-scaled)
        score_perf = clamp01((100 - dt_step_ms) / 90)     # good=10ms, bad=100ms


        # -------------------------------
        # 4. Category Scores (Weighted)
        # -------------------------------

        SafetyScore = (
            0.45 * score_rho_max +
            0.25 * score_rho_over +
            0.30 * score_collisions
        )

        EfficiencyScore = (
            0.55 * score_T_total +
            0.45 * score_t90
        )

        ComfortScore = (
            0.50 * score_detour +
            0.50 * score_d_min
        )

        PerformanceScore = score_perf


        # -------------------------------
        # 5. FINAL OVERALL SCORE
        # -------------------------------
        OverallScore = (
            0.45 * SafetyScore +
            0.35 * EfficiencyScore +
            0.15 * ComfortScore +
            0.05 * PerformanceScore
        )


        # -------------------------------
        # Return all metrics and scores
        # -------------------------------
        return {
            "metrics": {
                "T_total": T_total,
                "t_90": t_90,
                "T_mean": T_mean,
                "T_95": T_95,
                "rho_max": rho_max,
                "rho_over": rho_over,
                "d_min_avg": d_min_avg,
                "N_collisions": N_collisions,
                "D_mean": D_mean,
                "dt_step": dt_step_avg
            },
            "scores": {
                "Safety": SafetyScore,
                "Efficiency": EfficiencyScore,
                "Comfort": ComfortScore,
                "Performance": PerformanceScore,
                "Overall": OverallScore
            }
        }


    def plot_dashboard(self, scores=None, save_path=None):
        print("Generating Analysis Dashboard...")
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3)

        # ... (Existing plots 1-7 remain same) ...
        # 1. Evacuation Curve (Exited Count vs Time)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.times, self.exited_counts, 'g-', lw=2)
        ax1.set_title('Evacuation Progress')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Exited Agents')
        ax1.grid(True)
        
        # Calculate T_total and t_90
        total_agents = self.exited_counts[-1] if self.exited_counts else 0
        if total_agents > 0:
            t_90_idx = next((i for i, x in enumerate(self.exited_counts) if x >= 0.9 * total_agents), -1)
            t_90 = self.times[t_90_idx] if t_90_idx != -1 else 0
            ax1.axvline(t_90, color='r', linestyle='--', label=f't_90: {t_90:.1f}s')
            ax1.legend()

        # 2. Average Speed vs Time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.times, self.avg_speeds, 'b-', lw=2)
        ax2.set_title('Average Crowd Speed')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.grid(True)

        # 3. Max Density vs Time
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.times, self.max_densities, 'r-', lw=2)
        ax3.set_title('Max Local Density')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Agents / Cell')
        ax3.grid(True)

        # 4. Collisions per Frame
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.times, self.collision_counts, 'k-', lw=1)
        ax4.set_title('Collisions / Overlaps')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Count')
        ax4.grid(True)

        # 5. Average Min Distance
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.times, self.avg_min_dists, 'm-', lw=2)
        ax5.set_title('Avg Min Neighbor Distance')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Distance (m)')
        ax5.grid(True)

        # 6. Travel Time Distribution (Histogram)
        ax6 = fig.add_subplot(gs[1, 2])
        valid_times = [t for t in self.agent_exit_times if t > 0]
        if valid_times:
            ax6.hist(valid_times, bins=20, color='skyblue', edgecolor='black')
            mean_time = np.mean(valid_times)
            p95_time = np.percentile(valid_times, 95)
            ax6.axvline(mean_time, color='b', linestyle='--', label=f'Mean: {mean_time:.1f}s')
            ax6.axvline(p95_time, color='r', linestyle='--', label=f'95%: {p95_time:.1f}s')
            ax6.legend()
        ax6.set_title('Travel Time Distribution')
        ax6.set_xlabel('Time (s)')

        # 7. Travel Distance Distribution (Histogram)
        ax7 = fig.add_subplot(gs[2, 0])
        valid_dists = [d for d in self.agent_travel_dists if d > 0]
        if valid_dists:
            ax7.hist(valid_dists, bins=20, color='orange', edgecolor='black')
            mean_dist = np.mean(valid_dists)
            ax7.axvline(mean_dist, color='b', linestyle='--', label=f'Mean: {mean_dist:.1f}m')
            ax7.legend()
        ax7.set_title('Travel Distance Distribution')
        ax7.set_xlabel('Distance (m)')

        # 8. Score Report
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        if scores:
            s = scores['scores']
            m = scores['metrics']
            report = (
                f"FLOOR PLAN SCORE: {s['Overall']*100:.1f} / 100\n"
                f"--------------------------------------\n"
                f"Safety:      {s['Safety']*100:.1f}  (Collisions: {m['N_collisions']}, Max Rho: {m['rho_max']:.1f})\n"
                f"Efficiency:  {s['Efficiency']*100:.1f}  (T_total: {m['T_total']:.1f}s, t_90: {m['t_90']:.1f}s)\n"
                f"Comfort:     {s['Comfort']*100:.1f}  (Detour: {m['D_mean']:.2f}x, Min Dist: {m['d_min_avg']:.2f}m)\n"
                f"Performance: {s['Performance']*100:.1f}  (Step Time: {m['dt_step']*1000:.1f}ms)\n"
            )
            ax8.text(0.1, 0.5, report, fontsize=16, va='center', family='monospace')
        else:
            ax8.text(0.1, 0.5, "No scores available", fontsize=14)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Dashboard saved to {save_path}")
            
        plt.show()
