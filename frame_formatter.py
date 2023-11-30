import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


class DepthImageConverter:
    def __init__(
        self, input_folder, output_folder="data/depth_frames", override_output=False
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.override_output = override_output
        self.depth_estimator = pipeline("depth-estimation")

    def convert_images(self):
        # Clear the output folder if override_output is True
        if self.override_output:
            self.clear_output_folder()

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Get a list of image files in the input folder, ordered by name
        image_files = sorted(
            [
                f
                for f in os.listdir(self.input_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        for image_file in tqdm(image_files, desc="Formatting frames", unit="frame"):
            input_path = os.path.join(self.input_folder, image_file)
            output_path = os.path.join(self.output_folder, f"depth_{image_file}")

            # Read the RGB image
            rgb_image = Image.open(input_path)

            # Convert the RGB image to a depth image (grayscale)
            depth_image = self.rgb_to_depth(rgb_image)

            # Save the depth image
            depth_image.save(output_path)

    def rgb_to_depth(self, image):
        # return depth_image
        image = self.depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        return image

    def clear_output_folder(self):
        # Clear the contents of the output folder using shutil.rmtree
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder)


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Convert RGB images to depth images.")
    parser.add_argument(
        "-i", "--input_folder", help="Path to the input folder containing RGB images."
    )
    parser.add_argument(
        "--output_folder",
        default="data/depth_frames",
        help="Path to the output folder for depth images. Default is 'data/depth_frames'.",
    )
    parser.add_argument(
        "--override_output",
        action="store_true",
        help="Override the output folder by deleting its contents.",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Create an instance of DepthImageConverter and call the convert_images method
    depth_converter = DepthImageConverter(
        args.input_folder, args.output_folder, args.override_output
    )
    depth_converter.convert_images()
