import os
import shutil
import sys
import argparse


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


def move_gt_images(source_directory, target_directory):
    # Ensure both directories exist
    if not os.path.isdir(source_directory):
        print(f"Error: Source directory '{source_directory}' does not exist.")
        return
    
    if not os.path.isdir(target_directory):
        # Create the target directory if it doesn't exist
        os.makedirs(target_directory)
        print(f"Created target directory: {target_directory}")
    
    # Find all files with "_gt" in their name
    gt_files_found = False
    for filename in os.listdir(source_directory):
        if "_gt" in filename and os.path.isfile(os.path.join(source_directory, filename)):
            gt_files_found = True
            # Get the new filename without "_gt"
            new_filename = filename.replace("_gt", "")
            
            # Source and destination paths
            source_path = os.path.join(source_directory, filename)
            target_path = os.path.join(target_directory, new_filename)
            
            # Move and rename the file
            shutil.move(source_path, target_path)
            print(f"Moved and renamed: {source_path} -> {target_path}")
    
    if not gt_files_found:
        print(f"No files with '_gt' in their name found in {source_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and organize image files.')
    parser.add_argument('source_dir', help='Source directory containing the images')
    parser.add_argument('target_dir', help='Target directory where _gt images will be moved')
    parser.add_argument('--stage1', action='store_true', help='Run stage 1: Extract images from subdirectories')
    parser.add_argument('--stage2', action='store_true', help='Run stage 2: Move _gt images to target directory')
    
    args = parser.parse_args()
    
    # If no specific stage is selected, run both stages
    if not args.stage1 and not args.stage2:
        args.stage1 = True
        args.stage2 = True
    
    # Stage 1: Rename and extract images from subdirectories
    if args.stage1:
        print("Running Stage 1: Extracting images from subdirectories...")
        rename_and_extract_images(args.source_dir)
    
    # Stage 2: Move "_gt" images to target directory and rename them
    if args.stage2:
        print("Running Stage 2: Moving _gt images to target directory...")
        move_gt_images(args.source_dir, args.target_dir)
