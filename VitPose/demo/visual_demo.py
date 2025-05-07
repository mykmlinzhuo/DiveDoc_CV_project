import os
import glob
import cv2
import numpy as np
import re

def natural_sort_key(path):
    """
    Generate a key for natural sorting by extracting numbers from the file name.
    """
    if path is None:
        return float('inf')  # Place None values at the end
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(path))]

def create_attracting_image(folder_paths, output_path, image_size=(455, 256), border_size=3, gap_size=5, title_height=60):
    """
    Create an image grid with 4x5 subimages using OpenCV.
    Each column comes from the same folder, and each row shares the same file index.

    Args:
        folder_paths (list): List of folder paths (one for each column).
        output_path (str): Path to save the resulting image.
        image_size (tuple): Size of each subimage (width, height).
        border_size (int): Size of the border around each subimage.
        gap_size (int): Gap size between subimages.
        title_height (int): Height of the title area above each column.
    """
    num_rows = 4
    num_cols = len(folder_paths)
    frames_gap = 17
    demo_frame = [34,51,68,102]
    # Calculate grid dimensions
    grid_width = num_cols * (image_size[0] + 2 * border_size + gap_size) - gap_size
    grid_height = num_rows * (image_size[1] + 2 * border_size + gap_size) - gap_size + title_height

    # Create a blank canvas for the grid
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    grid_image[:] = (20, 20, 20)  # Darker gray background

    # Add column titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    text_color = (255, 255, 255)  # White text

    for col, folder in enumerate(folder_paths):
        title = os.path.basename(folder)  # Use folder name as title
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        text_width, text_height = text_size
        x_offset = col * (image_size[0] + 2 * border_size + gap_size) + (image_size[0] + 2 * border_size - text_width) // 2
        y_offset = (title_height + text_height) // 2
        cv2.putText(grid_image, title, (x_offset, y_offset), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

    # Add images to the grid
    for col, folder in enumerate(folder_paths):
        image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")), key=natural_sort_key)

        # Ensure the list is consecutive with None for missing indices
        max_index = max([
            int(re.search(r'(\d+)(?=\.jpg$)', os.path.basename(p)).group(1)) 
            for p in image_paths if p is not None
        ])
        print(f"Max index for folder {folder}: {max_index}")
        consecutive_paths = [None] * (max_index + 1)
        for path in image_paths:
            if path is not None:
                index = int(re.search(r'(\d+)(?=\.jpg$)', os.path.basename(path)).group(1))
                consecutive_paths[index] = path

        if len(consecutive_paths) < num_rows:
            raise ValueError(f"Folder {folder} does not have enough images for {num_rows} rows.")

        for row, frame_index in enumerate(demo_frame):
            # if frames_gap * row >= len(consecutive_paths) or consecutive_paths[frames_gap * row] is None:
            #     continue
            # Read and resize the image
            if frame_index >= len(consecutive_paths) or consecutive_paths[frame_index] is None:
                print(f"Image not found for index {frame_index} in folder {folder}.")
                continue
            image = cv2.imread(consecutive_paths[frame_index])
            image = cv2.resize(image, image_size)

            # Add border around the image
            bordered_image = cv2.copyMakeBorder(
                image,
                border_size, border_size, border_size, border_size,
                cv2.BORDER_CONSTANT,
                value=(50, 50, 50)  # Darker border
            )

            # Calculate position in the grid
            x_offset = col * (image_size[0] + 2 * border_size + gap_size)
            y_offset = row * (image_size[1] + 2 * border_size + gap_size) + title_height

            # Paste the bordered image into the grid
            grid_image[y_offset:y_offset + bordered_image.shape[0], x_offset:x_offset + bordered_image.shape[1]] = bordered_image

    # Save the resulting grid image
    cv2.imwrite(output_path, grid_image)
    print(f"Grid image saved to {output_path}")

# Example usage
folder_paths = [
    "demo_data/original_data",
    "demo_data/masked_data",
    "demo_data/detected_data",
    "demo_data/pose_data",
]
output_path = "attracting_image.jpg"
create_attracting_image(folder_paths, output_path)