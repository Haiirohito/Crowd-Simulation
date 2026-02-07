import os
import time
import taichi as ti
import numpy as np
from datetime import datetime


class SimulationRecorder:
    def __init__(self, base_path="results"):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(base_path, self.timestamp)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print(f"Recording results to: {self.output_dir}")

        self.video_manager = ti.tools.VideoManager(
            output_dir=self.output_dir, framerate=30, automatic_build=False
        )
        self.frame_count = 0

        self.density_frames = []
        self.velocity_frames = []

    def capture_frame(self, img_field):
        """
        Captures the current frame from the Taichi field.
        img_field: ti.Vector.field(3, float)
        """
        self.video_manager.write_frame(img_field)
        self.frame_count += 1

    def record_density(self, density_field):
        """
        Records the current state of the density field.
        density_field: ti.field
        """
        d_np = density_field.to_numpy()
        self.density_frames.append(d_np)

    def record_velocity(self, velocity_field):
        """
        Records the current state of the velocity field.
        velocity_field: ti.Vector.field(2)
        """
        v_np = velocity_field.to_numpy()
        self.velocity_frames.append(v_np)

    def save_maze_image(self, img_field, filename="maze_layout.png"):
        """
        Saves a single image of the maze.
        """
        path = os.path.join(self.output_dir, filename)
        ti.tools.imwrite(img_field, path)
        print(f"Saved maze image to {path}")

    def finish(self):
        """
        Finalizes the recording and generates the MP4.
        Also saves the density data.
        """
        print(f"Exporting video with {self.frame_count} frames...")
        try:
            self.video_manager.make_video(gif=False, mp4=True)
            print("Video saved successfully.")
        except Exception as e:
            print(f"Failed to create MP4 video (ffmpeg might be missing): {e}")
            print("Frames are saved in the output directory.")

        if self.density_frames:
            print(f"Saving {len(self.density_frames)} frames of density data...")
            density_array = np.array(self.density_frames)
            npy_path = os.path.join(self.output_dir, "density_data.npy")
            np.save(npy_path, density_array)
            print(f"Density data saved to {npy_path}")

        if self.velocity_frames:
            print(f"Saving {len(self.velocity_frames)} frames of velocity data...")
            velocity_array = np.array(self.velocity_frames)
            npy_path = os.path.join(self.output_dir, "velocity_data.npy")
            np.save(npy_path, velocity_array)
            print(f"Velocity data saved to {npy_path}")

    def get_output_dir(self):
        return self.output_dir
