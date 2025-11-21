import os
import time
import taichi as ti
from datetime import datetime

class SimulationRecorder:
    def __init__(self, base_path="results"):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(base_path, self.timestamp)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print(f"Recording results to: {self.output_dir}")
        
        # Initialize VideoManager
        # framerate=60 to match the simulation time step roughly, or 30 for standard video
        self.video_manager = ti.tools.VideoManager(output_dir=self.output_dir, framerate=30, automatic_build=False)
        self.frame_count = 0

    def capture_frame(self, img_field):
        """
        Captures the current frame from the Taichi field.
        img_field: ti.Vector.field(3, float)
        """
        self.video_manager.write_frame(img_field)
        self.frame_count += 1

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
        """
        print(f"Exporting video with {self.frame_count} frames...")
        try:
            self.video_manager.make_video(gif=False, mp4=True)
            print("Video saved successfully.")
        except Exception as e:
            print(f"Failed to create MP4 video (ffmpeg might be missing): {e}")
            print("Frames are saved in the output directory.")

    def get_output_dir(self):
        return self.output_dir
