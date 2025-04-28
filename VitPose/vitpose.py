import torch
import requests
import numpy as np

from PIL import Image

from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, AutoProcessor, AutoModelForObjectDetection, GroundingDinoModel
import math
import cv2
from transformers.models.grounding_dino.modeling_grounding_dino import post_process
from ultralytics import YOLO


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

# url = "http://images.cocodataset.org/val2017/000000000139.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
for i in range(7, 84):
    image_path = "/localdata/syb/Released_FineDiving_Dataset/Dataset/FineDiving/01/1/0000"+ str(i+7378) + ".jpg"
    image = Image.open(image_path).convert("RGB")

    # 掩码图像路径
    mask_path = "/localdata/syb/Released_FineDiving_Dataset/Dataset/FineDiving_HM/01/1/000"+ f"{i:02d}"+".jpg"
    mask = Image.open(mask_path).convert("L")  # 转为灰度图

    # 将图像和掩码转换为 NumPy 数组
    image_np = np.array(image)
    mask_np = np.array(mask)

    # 确保掩码的尺寸与原始图像一致
    print(f"Original image size: {image.size}, mask size: {mask.size}")
    if mask_np.shape != image_np.shape[:2]:
        print("Resizing mask to match image dimensions...")
        mask = mask.resize(image.size, Image.NEAREST)
        mask_np = np.array(mask)

    # 应用掩码：将掩码区域设置为黑色
    masked_image_np = image_np.copy()
    masked_image_np[mask_np == 0] = 0  # 掩码值为 0 的区域变为黑色

    # 将结果转换回 PIL 图像
    image = Image.fromarray(masked_image_np)

    # ------------------------------------------------------------------------
    # Stage 1. Detect humans on the image
    # ------------------------------------------------------------------------

    # You can choose any detector of your choice
    # person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    # person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)
    # person_image_processor = AutoProcessor.from_pretrained("keremberke/yolov8m-coco")
    # person_model = AutoModelForObjectDetection.from_pretrained("keremberke/yolov8m-coco").to(device)

    model = YOLO("yolov8n.pt")
    print("Inputs keys:", inputs.keys())  # 查看字典中的键
    print("Pixel values shape:", inputs["pixel_values"].shape)  # 检查 pixel_values 的形状
    print("Pixel values dtype:", inputs["pixel_values"].dtype)  # 检查数据类型
    pixel_values=inputs["pixel_values"][0].permute(1, 2, 0).cpu().numpy()
    pixel_values = (pixel_values * 255).astype(np.uint8) 
    inputs_image = Image.fromarray(pixel_values)
    inputs_image.save(f"inputs_image_{i}.jpg")
    with torch.no_grad():
        outputs = person_model(**inputs)

    # results = person_image_processor.post_process_object_detection(
    #     outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
    # )
    results = model()


    result = results[0]  # take first image results

    # Human label refers 0 index in COCO dataset
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    # ------------------------------------------------------------------------
    # Stage 2. Detect keypoints for each person found
    # ------------------------------------------------------------------------

    # image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    # model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

    # inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

    # with torch.no_grad():
    #     outputs = model(**inputs)

    # image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-base")
    # model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-base", device_map=device)
    image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge").to(device)
    inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

    dataset_index = torch.tensor([0], device=device) # must be a tensor of shape (batch_size,)

    with torch.no_grad():
        outputs = model(**inputs, dataset_index=dataset_index)

    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
    image_pose_result = pose_results[0]  # results for first image



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
    keypoint_colors = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

    numpy_image = np.array(image)

    for pose_result in image_pose_result:
        scores = np.array(pose_result["scores"])
        keypoints = np.array(pose_result["keypoints"])

        # draw each point on image
        draw_points(numpy_image, keypoints, scores, keypoint_colors, keypoint_score_threshold=0.3, radius=4, show_keypoint_weight=False)

        # draw links
        draw_links(numpy_image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold=0.3, thickness=1, show_keypoint_weight=False)

    pose_image = Image.fromarray(numpy_image)
    pose_image.save(f"pose_image_{i}_masked.jpg")
    print(f"Pose image saved as pose_image_{i}.jpg")