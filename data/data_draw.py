import cv2
import numpy as np
import os

def combine_images_for_actions(action_data, output_path, grid_color=(255, 255, 255), grid_width=5):
    """
    Combine images for multiple actions into a single visualization with white grids.
    
    Args:
        action_data: Dictionary where keys are action labels and values are lists of 6 image paths
                     Example: {"403B": [img1.png, img2.png, ...], "107C": [img1.png, img2.png, ...]}
        output_path: Path to save the final combined image
        grid_color: Color of the grid lines (default: white)
        grid_width: Width of the grid lines in pixels
    """
    actions = list(action_data.keys())
    num_actions = len(actions)
    
    # First, get dimensions by reading one image
    sample_img = cv2.imread(action_data[actions[0]][0])
    if sample_img is None:
        raise ValueError(f"Could not read sample image: {action_data[actions[0]][0]}")
    img_height, img_width = sample_img.shape[:2]
    new_img_height = int(img_height * 2 / 3)
    # Update sample_img to resized version for dimension calculation
    sample_img = cv2.resize(sample_img, (img_width, new_img_height))
    img_height = new_img_height
    img_height, img_width = sample_img.shape[:2]
    
    # Calculate dimensions for each row (one action with 6 images)
    row_width = 6 * img_width + 5 * grid_width  # 6 images with 5 grid lines between them
    
    # Calculate dimensions for the entire output image
    total_height = num_actions * img_height + (num_actions - 1) * grid_width  # rows with grid lines between
    side_margin = 110  # For action labels, increased for better spacing
    
    # Create blank canvas with white background
    output_img = np.ones((total_height, row_width + 2 * side_margin, 3), dtype=np.uint8) * 255
    colors = [(255, 182, 193), (230, 219, 255), (189, 252, 201)]
    # Add each action row
    for action_idx, action_label in enumerate(actions):
        image_paths = action_data[action_label]
        
        # Calculate vertical position
        y_start = action_idx * (img_height + grid_width)
        
        # Add horizontal images for this action
        for img_idx, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_width, img_height))
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            # Calculate horizontal position
            x_start = side_margin + img_idx * (img_width + grid_width)
            
            # Place image
            output_img[y_start:y_start + img_height, x_start:x_start + img_width] = img
        
        # Font settings for better appearance
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.2
        font_thickness = 3
        text_color = (0, 0, 0)  # Black text
        
        # # Left side: Action label with better positioning
        # text_size, _ = cv2.getTextSize(action_label, font, font_scale, font_thickness)
        # text_width, text_height = text_size
        y_text = y_start + (img_height) // 2
        # cv2.putText(output_img, action_label, 
        #            (30, y_text), 
        #            font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # Right side: Action label
        x_text_right = side_margin + row_width + 10

        cv2.rectangle(output_img,
        (x_text_right - 5, y_start),
        (x_text_right + 150, y_start + img_height),
        colors[action_idx], -1)

        cv2.putText(output_img, action_label, 
                   (x_text_right, y_text), 
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Add "PRED" label at top right of images
        pred_text = "PRED"
        pred_size, _ = cv2.getTextSize(pred_text, font, 0.8, 1)
        y_pred = y_start + img_height // 4
        x_pred = 10
        cv2.putText(output_img, pred_text, 
                   (x_pred, y_pred), 
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # Add "ORIG" label at bottom right of images
        orig_text = "ORIG"
        orig_size, _ = cv2.getTextSize(orig_text, font, 0.8, 1)
        y_orig = y_start + 3 * img_height // 4
        x_orig = 10

        cv2.putText(output_img, orig_text, 
                   (x_orig, y_orig), 
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # Draw horizontal grid lines (between action rows)
    for i in range(1, num_actions):
        y = i * (img_height + grid_width) - grid_width
        cv2.rectangle(output_img, 
                      (0, y), 
                      (row_width + 2 * side_margin, y + grid_width), 
                      grid_color, -1)
    
    # Save the final image
    cv2.imwrite(output_path, output_img)
    print(f"Combined visualization saved to {output_path}")

# Example usage
action_data = {
    "407c": ["407c/50.png", "407c/90.png", "407c/120.png", "407c/150.png", "407c/180.png", "407c/190.png"],
    "107b": ["107b/20.png", "107b/30.png", "107b/125.png", "107b/215.png", "107b/220.png", "107b/250.png"],
    "307c": ["307c/10.png", "307c/45.png", "307c/70.png", "307c/110.png", "307c/160.png", "307c/175.png"],
}

# Replace with your actual image paths
output_path = "action_visualization.png"
combine_images_for_actions(action_data, output_path)