## 2025.4.27

### Updates:
- **Added**: `VitPose` folder for keypoint estimation.
- **Instructions**:
  1. Follow the `requirements.txt` file to install the required environment.
  2. Run `dataset_pose.py` to generate the file `fine-grained_annotation_aqa_with_keypoints.pkl`.

### Details:
- The file `fine-grained_annotation_aqa_with_keypoints.pkl` is similar to the original `fine-grained_annotation_aqa.pkl`, with the following difference:
  - For each key, its value now includes an **additional sixth element**, which is a list.
  - The list has a length equal to the number of images for that key.
  - Each element in the list is either:
    - A **dictionary** (if a person is detected in the corresponding frame), or
    - **None** (if no person is detected in the frame).

### Dictionary Structure:
- If a person is detected, the dictionary contains the following keys:
  1. **`keypoints`**: The detected keypoints for the person.
  2. **`scores`**: The confidence scores for each keypoint.
  3. **`labels`**: The labels associated with the detected keypoints.
  4. **`bbox`**: The bounding box coordinates for the detected person.

## 2025.4.28
### Updates:
- **Refactored**: The total pipe for running on a single gpu.
- **Baseline Results**: 
  | Metric        | Value       |
  |---------------|-------------|
  | tIoU_5        | 0.995995    |
  | tIoU_75       | 0.889186    |
  | Correlation    | 0.922811    |
  | L2            | 46.747684   |
  | RL2           | 0.004289    |
  | IOU Score     | 0.090167    |
  | F1 Score      | 0.160546    |
  | F2 Score      | 0.110866    |
  | Accuracy      | 0.934730    |
  | Recall        | 0.140347    |