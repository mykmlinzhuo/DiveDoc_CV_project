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

2025.5.12
Pose Embedding Final:
Naive: 2
Weighted: 3
Final Embedding: 5
Results:
Naive:
[TEST] EPOCH: -1, Loss_aqa: 8596.299805, Loss_tas: 9.952188, Loss_mask: 58.478523
[TEST] EPOCH: -1, tIoU_5: 0.996774, tIoU_75: 0.958065
[TEST] EPOCH: -1, correlation: 0.891822, L2: 25.896963, RL2: 0.005634
[TEST] EPOCH: -1, IOU Score: 0.141680, F1 Score: 0.242823, F2 Score: 0.172409, Accuracy: 0.958526, Recall: 0.207946

Weighted:

[TEST] EPOCH: -1, Loss_aqa: 7833.480957, Loss_tas: 11.933262, Loss_mask: 58.720970
[TEST] EPOCH: -1, tIoU_5: 0.996774, tIoU_75: 0.935484
[TEST] EPOCH: -1, correlation: 0.897406, L2: 24.321457, RL2: 0.005291
[TEST] EPOCH: -1, IOU Score: 0.161848, F1 Score: 0.272085, F2 Score: 0.196624, Accuracy: 0.958923, Recall: 0.236343

Final:
[TEST] EPOCH: -1, Loss_aqa: 9196.884766, Loss_tas: 9.964624, Loss_mask: 59.446674
[TEST] EPOCH: -1, tIoU_5: 1.000000, tIoU_75: 0.954839
[TEST] EPOCH: -1, correlation: 0.892625, L2: 27.877840, RL2: 0.006065
[TEST] EPOCH: -1, IOU Score: 0.149444, F1 Score: 0.254091, F2 Score: 0.182265, Accuracy: 0.959037, Recall: 0.215797

[TEST] EPOCH: -1, Loss_aqa: 23738.371094, Loss_tas: 5.073991, Loss_mask: 29.042435
[TEST] EPOCH: -1, tIoU_5: 0.500000, tIoU_75: 0.496774
[TEST] EPOCH: -1, correlation: 0.890980, L2: 27.723163, RL2: 0.006031
[TEST] EPOCH: -1, IOU Score: 0.079242, F1 Score: 0.134440, F2 Score: 0.096672, Accuracy: 0.479756, Recall: 0.122152