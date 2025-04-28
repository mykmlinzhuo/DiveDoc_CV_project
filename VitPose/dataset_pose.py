import pickle
import glob
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import requests
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, AutoProcessor, AutoModelForObjectDetection, GroundingDinoModel
import math
import cv2

data_root = '/localdata/syb/Released_FineDiving_Dataset/Dataset/FineDiving'
label_path = '/localdata/syb/Released_FineDiving_Dataset/Annotations/fine-grained_annotation_aqa.pkl'
train_split = '/localdata/syb/Released_FineDiving_Dataset/Annotations/train_split.pkl'
test_split = '/localdata/syb/Released_FineDiving_Dataset/Annotations/test_split.pkl'
data_mask_root = '/localdata/syb/Released_FineDiving_Dataset/Dataset/FineDiving_HM'

label_save_path = '/localdata/syb/Released_FineDiving_Dataset/Annotations/fine-grained_annotation_aqa_with_keypoints.pkl'

with open(label_path,'rb') as f:
    label_data = pickle.load(f)
# for key, value in label_data.items():
#     print(f"Key: {key}, Value: {value}")


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a model
detection_model = YOLO("yolov8n.pt")
image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge").to(device)
dataset_index = torch.tensor([0], device=device) # must be a tensor of shape (batch_size,)


keypoint_edges = model.config.edges

palette = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ]
)

link_colors = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
keypoint_colors = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

def draw_points(image, keypoints, scores, pose_keypoint_color, keypoint_score_threshold, radius, show_keypoint_weight):
    if pose_keypoint_color is not None:
        assert len(pose_keypoint_color) == len(keypoints)
    for kid, (kpt, kpt_score) in enumerate(zip(keypoints, scores)):
        x_coord, y_coord = int(kpt[0]), int(kpt[1])
        if kpt_score > keypoint_score_threshold:
            color = tuple(int(c) for c in pose_keypoint_color[kid])
            if show_keypoint_weight:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)
                transparency = max(0, min(1, kpt_score))
                cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
            else:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)

def draw_links(image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold, thickness, show_keypoint_weight, stick_width = 2):
    height, width, _ = image.shape
    if keypoint_edges is not None and link_colors is not None:
        assert len(link_colors) == len(keypoint_edges)
        for sk_id, sk in enumerate(keypoint_edges):
            x1, y1, score1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]), scores[sk[0]])
            x2, y2, score2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]), scores[sk[1]])
            if (
                x1 > 0
                and x1 < width
                and y1 > 0
                and y1 < height
                and x2 > 0
                and x2 < width
                and y2 > 0
                and y2 < height
                and score1 > keypoint_score_threshold
                and score2 > keypoint_score_threshold
            ):
                color = tuple(int(c) for c in link_colors[sk_id])
                if show_keypoint_weight:
                    X = (x1, x2)
                    Y = (y1, y2)
                    mean_x = np.mean(X)
                    mean_y = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    polygon = cv2.ellipse2Poly(
                        (int(mean_x), int(mean_y)), (int(length / 2), int(stick_width)), int(angle), 0, 360, 1
                    )
                    cv2.fillConvexPoly(image, polygon, color)
                    transparency = max(0, min(1, 0.5 * (keypoints[sk[0], 2] + keypoints[sk[1], 2])))
                    cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
                else:
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)



with open(train_split, 'rb') as f:
    train_data = pickle.load(f)

with open(test_split, 'rb') as f:
    test_data = pickle.load(f)
print(f"type of train_data: {type(train_data)}")
print(f"type of test_data: {type(test_data)}")
all_data = list(set(train_data + test_data))
print(f"length of all_data: {len(all_data)}")
print(f"length of train_data: {len(train_data)}")
print(f"length of test_data: {len(test_data)}")
assert len(all_data) == len(train_data) + len(test_data), "All data length mismatch"

length_of_all_data = len(all_data)
for id, item in enumerate(all_data):
    print(f"Processing on Key: {item[0]}, Value: {item[1]}, ID: {id}, Total: {length_of_all_data}")
    image_list = sorted((glob.glob(os.path.join(data_root, item[0], str(item[1]), '*.jpg'))))
    mask_list = sorted((glob.glob(os.path.join(data_mask_root, item[0], str(item[1]), '*.jpg'))))
    assert len(image_list) == len(mask_list), f"Image and mask lists have different lengths for {item}"
    images = []
    for i in range(len(image_list)):
        image = Image.open(image_list[i])
        mask = Image.open(mask_list[i]).convert("L")
        image_np = np.array(image)
        mask_np = np.array(mask)

        if mask_np.shape != image_np.shape[:2]:
            mask = mask.resize(image.size, Image.NEAREST)
            mask_np = np.array(mask)
        masked_image_np = image_np.copy()
        masked_image_np[mask_np == 0] = 0  
        images.append(Image.fromarray(masked_image_np))
    results = detection_model.predict(
        source=images,
        classes=[0],  # Person class
        max_det=1,  # Only one person per image
        conf=0.1,  # Confidence threshold
    )
    keypoint_detection_results = []
    for idx, result in enumerate(results):
        # result.save(f"result_{item[0]}_{item[1]}_{idx}.jpg")
        # if idx ==48 and item == ('FINAWorldChampionships2019_Women10m_final_r1', 1):
        #     result.save(f"result_{item[0]}_{item[1]}_{idx}.jpg")
        person_boxes = result.boxes.xyxy.cpu().numpy()
        if person_boxes.shape[0] == 0:
            print(f"No person detected in image {idx} of video {item[0]}_{item[1]}")
            keypoint_detection_results.append(None)
            continue
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
        inputs = image_processor(images[idx], boxes=[person_boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, dataset_index=dataset_index)
        pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
        image_pose_result = pose_results[0]  # results for first image
        # print(f"Pose results for image {idx} of video {item[0]}_{item[1]}: {image_pose_result}")
        # Save the pose results
        keypoint_detection_results.append(image_pose_result[0])
        assert type(image_pose_result[0]) == dict, f"Keypoint detection result is not a dictionary for {item} at index {idx}"
        assert len(image_pose_result) == 1, f"Keypoint detection result length is not 1 for {item} at index {idx}"
        
        # numpy_image = np.array(images[idx])
        # scores = np.array(image_pose_result[0]["scores"])
        # keypoints = np.array(image_pose_result[0]["keypoints"])

        # # draw each point on image
        # draw_points(numpy_image, keypoints, scores, keypoint_colors, keypoint_score_threshold=0, radius=4, show_keypoint_weight=False)

        # # draw links
        # draw_links(numpy_image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold=0, thickness=1, show_keypoint_weight=False)

        # pose_image = Image.fromarray(numpy_image)
        # pose_image.save(f"result_pose_image_{item[0]}_{item[1]}_{idx}.jpg")

    print(f"length of keypoint_detection_results: {len(keypoint_detection_results)}")
    print(f"length of images: {len(images)}")
    assert len(keypoint_detection_results) == len(images), f"Keypoint detection results length mismatch for {item}"
    label_data[item].append(keypoint_detection_results)
    assert len(label_data[item]) == 6, f"Label data length mismatch for {item}"


with open(label_save_path, 'wb') as f:
    pickle.dump(label_data, f)






