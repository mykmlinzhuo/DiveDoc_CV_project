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
  | tIoU_75       | 0.687500    |
  | Correlation    |  0.850626   |
  | RL2           | 0.014540   |
  | IOU Score     | 0.090167    |
  | F1 Score      | 0.035744    |
  | F2 Score      | 0.023285    |
  | Accuracy      | 0.952926    |
  | Recall        | 0.051676    |

## 2025.5.12
### Pose Embedding Final:
- Naive: 2
- Weighted: 3
- Final Embedding: 5

### Results:
Naive: 
- [TEST] EPOCH: -1, Loss_aqa: 23665.031250, Loss_tas: 4.978251, Loss_mask: 29.113617

- [TEST] EPOCH: -1, tIoU_5: 0.500000, tIoU_75: 0.496774

- [TEST] EPOCH: -1, correlation: 0.894333, L2: 25.647221, RL2: 0.005579

- [TEST] EPOCH: -1, IOU Score: 0.067038, F1 Score: 0.116243, F2 Score: 0.081758, Accuracy: 0.479295, Recall: 0.104875

Weighted:
- [TEST] EPOCH: -1, Loss_aqa: 21594.777344, Loss_tas: 5.978729, Loss_mask: 29.210537

- [TEST] EPOCH: -1, tIoU_5: 0.500000, tIoU_75: 0.490323
  
- [TEST] EPOCH: -1, correlation: 0.898070, L2: 24.363815, RL2: 0.005300

- [TEST] EPOCH: -1, IOU Score: 0.076103, F1 Score: 0.129787, F2 Score: 0.092725, Accuracy: 0.479598, Recall: 0.118205

Final:
- [TEST] EPOCH: -1, Loss_aqa: 21622.794922, Loss_tas: 5.006417, Loss_mask: 29.867155

- [TEST] EPOCH: -1, tIoU_5: 0.500000, tIoU_75: 0.490323

- [TEST] EPOCH: -1, correlation: 0.892136, L2: 27.750083, RL2: 0.006037

- [TEST] EPOCH: -1, IOU Score: 0.070353, F1 Score: 0.121314, F2 Score: 0.086036, Accuracy: 0.479370, Recall: 0.108345

