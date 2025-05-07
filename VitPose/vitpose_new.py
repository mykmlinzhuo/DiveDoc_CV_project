from ultralytics import YOLO
import torch
import requests
import numpy as np

from PIL import Image

from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, AutoProcessor, AutoModelForObjectDetection, GroundingDinoModel
import math
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


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


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a model
model = YOLO("yolov8n.pt")

# Perform object detection on an image
image = Image.open("/localdata/syb/Released_FineDiving_Dataset/Dataset/FineDiving/01/1/00007386.jpg")
mask = Image.open("/localdata/syb/Released_FineDiving_Dataset/Dataset/FineDiving_HM/01/1/00008.jpg").convert("L")
image_np = np.array(image)
mask_np = np.array(mask)

# 确保掩码的尺寸与原始图像一致
print(f"Original image size: {image.size}, mask size: {mask.size}")
if mask_np.shape != image_np.shape[:2]:
    print("Resizing mask to match image dimensions...")
    mask = mask.resize(image.size, Image.NEAREST)
    mask_np = np.array(mask)
masked_image_np = image_np.copy()
masked_image_np[mask_np == 0] = 0  # 掩码值为 0 的区域变为黑色

# 将结果转换回 PIL 图像
image = Image.fromarray(masked_image_np)
results = model.predict([image])
results[0].save("results.jpg")  # save results to 'runs/detect/predict' directory
print("Detection Results:")
# for i, box in enumerate(results[0].boxes):
#     cls = int(box.cls[0])  # Class index
#     conf = float(box.conf[0])  # Confidence score
#     xyxy = box.xyxy[0]  # Bounding box coordinates (x1, y1, x2, y2)
#     print(f"Detection {i + 1}:")
#     print(f"  Class: {cls}")
#     print(f"  Confidence: {conf:.2f}")
#     print(f"  Bounding Box: {xyxy}")
#     print(f" Type of bounding box: {type(xyxy)}")
person_boxes = results[0].boxes.xyxy.cpu().numpy()
print(f"Person boxes: {person_boxes}")
print(f"shape of person boxes: {person_boxes.shape}")
person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge").to(device)
inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

dataset_index = torch.tensor([0], device=device) # must be a tensor of shape (batch_size,)

with torch.no_grad():
    outputs = model(**inputs, dataset_index=dataset_index)

pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
image_pose_result = pose_results[0]  # results for first image

print(f"Pose results: {image_pose_result}")
print(f"Pose results keys: {image_pose_result[0].keys()}")
print(f"Pose results scores: {image_pose_result[0]['scores']}")

# Note: keypoint_edges and color palette are dataset-specific
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
keypoint_colors = palette[[17, 17, 17, 17, 17, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

numpy_image = np.array(image)

for pose_result in image_pose_result:
    scores = np.array(pose_result["scores"])
    print(f"type of scores: {type(pose_result['scores'])}")
    keypoints = np.array(pose_result["keypoints"])

    # draw each point on image
    draw_points(numpy_image, keypoints, scores, keypoint_colors, keypoint_score_threshold=0, radius=4, show_keypoint_weight=False)

    # draw links
    draw_links(numpy_image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold=0, thickness=1, show_keypoint_weight=False)

pose_image = Image.fromarray(numpy_image)
pose_image.save(f"pose_image.jpg")
print(f"Pose image saved as pose_image.jpg")