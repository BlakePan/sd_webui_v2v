import argparse
import datetime
import os
import shutil

import cv2
from tqdm import tqdm


class VideoExtractor:
    def __init__(
        self,
        input_file,
        output_folder="data/output_frames",
        fps=1,
        file_extension="png",
        override_output=False,
        start_time="00:00:00",
        end_time=None,
        first_n_mins=None,
    ):
        self.input_file = input_file
        self.output_folder = output_folder
        self.fps = fps
        self.file_extension = file_extension
        self.override_output = override_output
        self.start_time = self.parse_time(start_time)
        self.end_time = self.parse_time(end_time) if end_time else None

        # Calculate frames per second of the original video
        self.video_fps = self.get_video_fps()
        self.first_n_mins = first_n_mins
        self.first_n_frames = (
            int(first_n_mins * 60 * self.video_fps) if first_n_mins else None
        )

    def extract_frames(self):
        # Check if the input file exists
        if not os.path.exists(self.input_file):
            print(f"Error: Input file '{self.input_file}' not found.")
            return

        # Check if the input file is a valid video file
        if not self.is_valid_video():
            print(f"Error: '{self.input_file}' is not a valid video file.")
            return

        # Clear the output folder if override_output is True
        if self.override_output:
            self.clear_output_folder()

        # Open the video file
        cap = cv2.VideoCapture(self.input_file)

        # Calculate the frame interval based on the desired fps
        frame_interval = int(self.video_fps / self.fps)

        # Calculate frame indices for start and end times
        start_frame_index = int(self.cal_total_seconds(self.start_time) * self.video_fps)
        end_frame_index = None
        if self.end_time:
            end_frame_index = int(self.cal_total_seconds(self.end_time) * self.video_fps)

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Read and save frames
        frame_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use tqdm to create a progress bar
        with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()

                if (
                    not ret
                    or (end_frame_index is not None and frame_index >= end_frame_index)
                    or (
                        self.first_n_frames
                        and frame_index >= start_frame_index + self.first_n_frames
                    )
                ):
                    break

                # Save frame if it's within the specified time range
                if (
                    frame_index % frame_interval == 0
                    and frame_index >= start_frame_index
                    and (
                        self.first_n_frames is None
                        or frame_index < start_frame_index + self.first_n_frames
                    )
                ):
                    output_path = os.path.join(
                        self.output_folder,
                        f"frame_{frame_index:04d}.{self.file_extension}",
                    )
                    cv2.imwrite(output_path, frame)

                frame_index += 1
                pbar.update(1)  # Update the progress bar

        # Release the video capture object
        cap.release()

    def clear_output_folder(self):
        # Clear the contents of the output folder using shutil.rmtree
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder)

    def cal_total_seconds(self, timeobj):
        return (
            datetime.datetime.combine(datetime.date.min, timeobj)
            - datetime.datetime.min
        ).total_seconds()

    def parse_time(self, time_str):
        try:
            return datetime.datetime.strptime(time_str, "%H:%M:%S").time()
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid time format. Use HH:MM:SS.")

    def is_valid_video(self):
        # Check if the input file is a valid video file
        try:
            cap = cv2.VideoCapture(self.input_file)
            return cap.isOpened()
        except Exception as e:
            return False

    def get_video_fps(self):
        # Get the frames per second (fps) of the input video
        cap = cv2.VideoCapture(self.input_file)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return video_fps


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("-i", "--input_file", help="Path to the input video file.")
    parser.add_argument(
        "--output_folder",
        default="data/output_frames",
        help="Path to the output folder. Default is 'output_frames'.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1,
        help="Frames per second to extract. Default is 1.",
    )
    parser.add_argument(
        "--file_extension",
        default="png",
        help="File extension for the output frames. Default is 'png'.",
    )
    parser.add_argument(
        "--override_output",
        action="store_true",
        help="Override the output folder by deleting its contents.",
    )
    parser.add_argument(
        "-s",
        "--start_time",
        default="00:00:00",
        help="Start time for extracting frames. Format is HH:MM:SS. Default is '00:00:00'.",
    )
    parser.add_argument(
        "-e", "--end_time", help="End time for extracting frames. Format is HH:MM:SS."
    )
    parser.add_argument(
        "-m",
        "--first_n_mins",
        type=float,
        help="Extract frames for the first n minutes of the video.",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Create an instance of VideoExtractor and call the extract_frames method
    video_extractor = VideoExtractor(
        args.input_file,
        args.output_folder,
        args.fps,
        args.file_extension,
        args.override_output,
        args.start_time,
        args.end_time,
        args.first_n_mins,
    )
    video_extractor.extract_frames()
