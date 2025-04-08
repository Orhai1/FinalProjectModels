import os
import subprocess
import numpy as np
import cv2
import json
from skimage.metrics import structural_similarity as ssim


class VideoFrameExtractor:
    def __init__(self, input_dir, output_dir):
        """
        Initialize the video frame extractor

        :param input_dir: Directory containing input videos
        :param output_dir: Directory to save extracted frames
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def extract_scene_change_frames(self,
                                    video_path,
                                    scene_change_threshold=0.01,
                                    fps=30):
        """
        Extract frames at scene changes with high-frequency sampling

        :param video_path: Path to the input video
        :param scene_change_threshold: Sensitivity for scene change detection
        :param fps: Frames per second to extract
        :return: List of extracted frame paths
        """
        # Temporary directory for initial frame extraction
        temp_frame_dir = os.path.join(self.output_dir,
                                      f"temp_frames_{os.path.basename(video_path)}")
        os.makedirs(temp_frame_dir, exist_ok=True)

        # Extract frames using FFmpeg
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={fps}',  # Extract at specified frames per second
            f'{temp_frame_dir}/frame_%05d.png'
        ]

        subprocess.run(ffmpeg_cmd, capture_output=True)

        # Get list of extracted frames
        frames = sorted([os.path.join(temp_frame_dir, f) for f in os.listdir(temp_frame_dir)
                         if f.endswith('.png')])

        # Scene change detection
        scene_change_frames = []
        prev_frame = None

        for i, frame_path in enumerate(frames):
            current_frame = cv2.imread(frame_path)

            if prev_frame is not None:
                # Convert to grayscale for comparison
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                # Resize frames to reduce computation and improve consistency
                h, w = min(prev_gray.shape[0], current_gray.shape[0]), \
                    min(prev_gray.shape[1], current_gray.shape[1])
                prev_gray = cv2.resize(prev_gray, (w, h))
                current_gray = cv2.resize(current_gray, (w, h))

                # Compute structural similarity with multiple methods
                try:
                    # SSIM method
                    ssim_score = ssim(prev_gray, current_gray)

                    # Absolute difference method
                    diff = cv2.absdiff(prev_gray, current_gray)
                    diff_score = np.mean(diff)

                    # Combined detection
                    if (1 - ssim_score > scene_change_threshold or
                            diff_score > scene_change_threshold * 255):
                        output_frame_path = os.path.join(
                            self.output_dir,
                            f"{os.path.basename(video_path)}_scene_{i:05d}.png"
                        )
                        cv2.imwrite(output_frame_path, current_frame)
                        scene_change_frames.append(output_frame_path)
                except Exception as e:
                    print(f"Error comparing frames: {e}")

            prev_frame = current_frame

        # Clean up temporary frames
        self._cleanup_temp_dir(temp_frame_dir)

        return scene_change_frames

    def _cleanup_temp_dir(self, temp_dir):
        """
        Clean up temporary frame directory

        :param temp_dir: Path to temporary directory
        """
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

    def process_video_directory(self,
                                scene_change_threshold=0.01,
                                fps=30):
        """
        Process all videos in the input directory

        :param scene_change_threshold: Sensitivity for scene change detection
        :param fps: Frames per second to extract
        :return: Dictionary of video paths and their scene change frames
        """
        video_scene_frames = {}

        # Supported video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

        for filename in os.listdir(self.input_dir):
            # Check if file is a video
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(self.input_dir, filename)

                try:
                    scene_frames = self.extract_scene_change_frames(
                        video_path,
                        scene_change_threshold=scene_change_threshold,
                        fps=fps
                    )
                    video_scene_frames[video_path] = scene_frames
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        return video_scene_frames


# Example usage
def main():
    input_video_dir = 'videos'
    output_frame_dir = 'output_frames2'

    extractor = VideoFrameExtractor(input_video_dir, output_frame_dir)

    # Process all videos in directory
    scene_change_results = extractor.process_video_directory(
        scene_change_threshold=0.01,  # Lower threshold for more sensitive detection
        fps=30  # Extract 30 frames per second
    )

    # Optional: Save results to a JSON file
    with open('../scene_change_results.json', 'w') as f:
        json.dump({k: v for k, v in scene_change_results.items()}, f, indent=2)


if __name__ == '__main__':
    main()