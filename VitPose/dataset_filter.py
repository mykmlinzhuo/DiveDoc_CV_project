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

TOP_K = 5

data_root = '/localdata/syb/Released_FineDiving_Dataset/Dataset/FineDiving'
label_path = '/localdata/syb/Released_FineDiving_Dataset/Annotations/fine-grained_annotation_aqa.pkl'
train_split = '/localdata/syb/Released_FineDiving_Dataset/Annotations/train_split.pkl'
test_split = '/localdata/syb/Released_FineDiving_Dataset/Annotations/test_split.pkl'
data_mask_root = '/localdata/syb/Released_FineDiving_Dataset/Dataset/FineDiving_HM'

new_train_split = '/localdata/syb/Released_FineDiving_Dataset/Annotations/train_split_' + str(TOP_K) + '.pkl'
new_test_split = '/localdata/syb/Released_FineDiving_Dataset/Annotations/test_split_' + str(TOP_K) + '.pkl'

# with open(label_path,'rb') as f:
#     label_data = pickle.load(f)
# for key, value in label_data.items():
#     print(f"Key: {key}, Value: {value[0]}")

with open(label_path, 'rb') as f:
    label_data = pickle.load(f)

# Count occurrences of each action type and store associated keys
action_counts = {}
action_keys = {}  # Dictionary to store keys for each action type

for key, value in label_data.items():
    action_type = value[0]
    # Count occurrences
    if action_type in action_counts:
        action_counts[action_type] += 1
        action_keys[action_type].append(key)  # Append key to the list
    else:
        action_counts[action_type] = 1
        action_keys[action_type] = [key]  # Initialize list with the first key

# Sort action types by count in decreasing order
sorted_action_counts = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)

# Print the sorted action counts and associated keys
for action_type, count in sorted_action_counts:
    print(f"Action Type: {action_type}, Count: {count}")
    # print(f"Keys: {action_keys[action_type]}")
top_k_action_types = [action_type for action_type, _ in sorted_action_counts[:TOP_K]]
# Print the top K action types
print(f"Top {TOP_K} Action Types: {top_k_action_types}")


with open(train_split, 'rb') as f:
    train_data = pickle.load(f)

with open(test_split, 'rb') as f:
    test_data = pickle.load(f)

# Filter train_data and test_data
def filter_data(data_keys, label_data, top_k_action_types):
    filtered_data = []
    for key in data_keys:
        if key in label_data and label_data[key][0] in top_k_action_types:
            filtered_data.append(key)
    return filtered_data

new_train_data = filter_data(train_data, label_data, top_k_action_types)
new_test_data = filter_data(test_data, label_data, top_k_action_types)


print(f"Original train data size: {len(train_data)}")
print(f"Filtered train data size: {len(new_train_data)}")
print(f"Original test data size: {len(test_data)}")
print(f"Filtered test data size: {len(new_test_data)}")

# # Save the new train and test data
# with open(new_train_split, 'wb') as f:
#     pickle.dump(new_train_data, f)

# with open(new_test_split, 'wb') as f:
#     pickle.dump(new_test_data, f)

# print(f"New train data saved to {new_train_split}")
# print(f"New test data saved to {new_test_split}")




