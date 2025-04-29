import pickle
import glob
import os
from PIL import Image
import numpy as np
import torch
import requests
import math
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

device = "cuda" if torch.cuda.is_available() else "cpu"

label_save_path = '/localdata/syb/Released_FineDiving_Dataset/Annotations/fine-grained_annotation_aqa_with_keypoints.pkl'
with open(label_save_path, 'rb') as f:
    label_data = pickle.load(f)
for key, value in label_data.items():
    print(f"Processing key: {key}")
    for element in value[5]:
        if element is None:
            continue  # 跳过 None 的元素
        assert len(element["keypoints"]) == 17 
        assert element['labels'].equal(torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]))
        assert len(element['scores']) == 17
        assert len(element['bbox']) == 4