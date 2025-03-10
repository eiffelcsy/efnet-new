import os
import shutil
import sys


def rename_and_extract_images(input_directory):
    # Ensure the input directory exists
    if not os.path.isdir(input_directory):
        print(f"Error: Directory '{input_directory}' does not exist.")
        return

    # Iterate over all subdirectories
    for subfolder in os.listdir(input_directory):
        subfolder_path = os.path.join(input_directory, subfolder)

        # Ensure it is a directory
        if os.path.isdir(subfolder_path):
            image_1 = os.path.join(subfolder_path, "0000.png")
            image_2 = os.path.join(subfolder_path, "0000_gt.png")

            # New filenames
            renamed_image_1 = os.path.join(input_directory, f"{subfolder}.png")
            renamed_image_2 = os.path.join(input_directory, f"{subfolder}_gt.png")

            # Rename and move images if they exist
            if os.path.exists(image_1):
                shutil.move(image_1, renamed_image_1)
                print(f"Renamed and moved: {image_1} -> {renamed_image_1}")

            if os.path.exists(image_2):
                shutil.move(image_2, renamed_image_2)
                print(f"Renamed and moved: {image_2} -> {renamed_image_2}")

            # Remove the empty subfolder
            shutil.rmtree(subfolder_path)
            print(f"Removed folder: {subfolder_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_path>")
    else:
        directory = sys.argv[1]
        rename_and_extract_images(directory)
