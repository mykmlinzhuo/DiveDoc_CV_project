import torch
import numpy as np
import os
import pickle
import random
import glob
from os.path import join
from PIL import Image

class FineDiving_Pair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform
        self.random_choosing = args.random_choosing
        self.action_number_choosing = args.action_number_choosing
        self.length = args.frame_length
        self.voter_number = args.voter_number
        self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
        
        # file path
        self.data_root = args.data_root
        self.mask_root = args.data_mask_root
        self.data_anno = self.read_pickle(args.label_path)
        self.data_anno_pose = self.data_anno
        with open(args.train_split, 'rb') as f:
            self.train_dataset_list = pickle.load(f)
        with open(args.test_split, 'rb') as f:
            self.test_dataset_list = pickle.load(f)

        self.action_number_dict = {}
        self.difficulties_dict = {}
        if self.subset == 'train':
            self.dataset = self.train_dataset_list
        else:
            self.dataset = self.test_dataset_list
            self.action_number_dict_test = {}
            self.difficulties_dict_test = {}

        self.choose_list = self.train_dataset_list.copy()
        if self.action_number_choosing:
            self.preprocess()
            self.check_exemplar_dict()

    def preprocess(self):
        for item in self.train_dataset_list:
            dive_number = self.data_anno.get(item)[0]
            if self.action_number_dict.get(dive_number) is None:
                self.action_number_dict[dive_number] = []
            self.action_number_dict[dive_number].append(item)
        if self.subset == 'test':
            for item in self.test_dataset_list:
                dive_number = self.data_anno.get(item)[0]
                if self.action_number_dict_test.get(dive_number) is None:
                    self.action_number_dict_test[dive_number] = []
                self.action_number_dict_test[dive_number].append(item)

    def check_exemplar_dict(self):
        if self.subset == 'train':
            for key in sorted(list(self.action_number_dict.keys())):
                file_list = self.action_number_dict[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key
        if self.subset == 'test':
            for key in sorted(list(self.action_number_dict_test.keys())):
                file_list = self.action_number_dict_test[key]
                for item in file_list:
                    assert self.data_anno[item][0] == key

    def process_pose_list(
        self,
        pose_list,
        length,
        num_joints: int = 17,
        score_threshold: float = 0.0,
    ):
        """
        把任意格式的 pose_list 规范成长度 = length 的 list[dict]，
        每个 dict 里 keypoints 必定是 (17,2) 的单帧张量。
        """
        processed = []
        default_labels = torch.arange(num_joints, dtype=torch.int64)

        def make_blank():
            return {
                "keypoints": torch.zeros((num_joints, 2), dtype=torch.float32),
                "scores":    torch.zeros((num_joints,),    dtype=torch.float32),
                "labels":    default_labels.clone(),
                "bbox":      torch.zeros((4,),            dtype=torch.float32),
            }

        for item in pose_list:
            if item is None:
                processed.append(make_blank())
                continue

            kpts = item["keypoints"]
            scores = item["scores"]
            labels = item["labels"]
            bbox = item["bbox"]

            # --- ❶ clip 情况：多帧 → 逐帧拆开 ---
            if kpts.dim() == 3:          # (N,17,2)
                N = kpts.shape[0]
                for n in range(N):
                    processed.append({
                        "keypoints": kpts[n].clone(),
                        "scores":    scores[n].clone(),
                        "labels":    labels[n].clone(),
                        "bbox":      bbox[n].clone(),
                    })
            else:                        # (17,2) 正常帧
                frame_pose = {
                    "keypoints": kpts.clone(),
                    "scores":    scores.clone(),
                    "labels":    labels.clone(),
                    "bbox":      bbox.clone(),
                }
                processed.append(frame_pose)

        # --- ❷ 低阈值置 0 ---
        if score_threshold > 0:
            for d in processed:
                mask = d["scores"] < score_threshold
                d["keypoints"][mask] = 0

        # --- ❸ 长度对齐 ---
        if len(processed) < length:
            processed.extend([make_blank() for _ in range(length - len(processed))])
        elif len(processed) > length:
            processed = processed[:length]

        return processed

    
    def load_video(self, video_file_name):
        image_list = sorted((glob.glob(os.path.join(self.data_root, video_file_name[0], str(video_file_name[1]), '*.jpg'))))
        mask_list = sorted((glob.glob(os.path.join(self.mask_root, video_file_name[0], str(video_file_name[1]), '*.jpg'))))

        
        start_frame = int(image_list[0].split("/")[-1][:-4])
        end_frame = int(image_list[-1].split("/")[-1][:-4])
        
        if self.subset == 'train':
            temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
            end_frame = end_frame + temporal_aug_shift
        
        frame_list = np.linspace(start_frame, end_frame, self.length).astype(int)
        image_frame_idx = [frame_list[i] - start_frame for i in range(self.length)]

        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.length)]
        masks = [Image.open(mask_list[image_frame_idx[i]]) for i in range(self.length)]

        frames_labels = [self.data_anno.get(video_file_name)[4][i] for i in image_frame_idx]
        frames_catogeries = list(set(frames_labels))
        frames_catogeries.sort(key=frames_labels.index)
        transitions = [frames_labels.index(c) for c in frames_catogeries]

        _video, _masks = self.transforms(video, masks)
        
        pose_detections = None
        # print(self.data_anno.get(video_file_name))
        if self.data_anno.get(video_file_name) is not None and len(self.data_anno.get(video_file_name)) >= 6:
            raw_pose_list = self.data_anno.get(video_file_name)[5]
            # print("="*40)
            # print(f"[DEBUG] raw_pose_list length: {len(raw_pose_list)}")
            pose_detections = []
            for idx in image_frame_idx:
                if idx < len(raw_pose_list):
                    # print(f"\n[DEBUG] Accessing frame_id {idx}:")
                    pose_info = raw_pose_list[idx]
                    # print(f"  type(pose_info): {type(pose_info)}")
                    # if isinstance(pose_info, dict):
                    #     for k, v in pose_info.items():
                    #         print(f"    key: {k}, type: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")
                    # else:
                    #     print(f"  pose_info is not a dict, type: {type(pose_info)}")
                    pose_detections.append(pose_info)
                else:
                    # print(f"\n[DEBUG] Missing pose for frame_id {idx}, append None")
                    pose_detections.append(None)
        else:
            pose_detections = [None for _ in range(self.length)]

        # 【统一处理成每帧都有标准化结构的pose字典】
        pose_detections = self.process_pose_list(pose_detections, self.length)
        return _video, _masks, np.array([transitions[1]-1,transitions[-1]-1]), np.array(frames_labels), pose_detections



    def read_pickle(self, pickle_path):
        with open(pickle_path,'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def __getitem__(self, index):
        sample_1  = self.dataset[index]
        data = {}
        data['video'], data['video_mask'], data['transits'], data['frame_labels'], data['pose_detections'] = self.load_video(sample_1)
        data['number'] = self.data_anno.get(sample_1)[0]
        data['final_score'] = self.data_anno.get(sample_1)[1]
        data['difficulty'] = self.data_anno.get(sample_1)[2]
        data['completeness'] = (data['final_score'] / data['difficulty'])

        # choose a exemplar
        if self.subset == 'train':
            # train phrase
            if self.action_number_choosing == True:
                file_list = self.action_number_dict[self.data_anno[sample_1][0]].copy()
            elif self.DD_choosing == True:
                file_list = self.difficulties_dict[self.data_anno[sample_1][2]].copy()
            else:
                # randomly
                file_list = self.train_dataset_list.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            target['video'], target['video_mask'], target['transits'], target['frame_labels'], target['pose_detections'] = self.load_video(sample_2)
            # import pdb; pdb.set_trace()
            target['number'] = self.data_anno.get(sample_2)[0]
            target['final_score'] = self.data_anno.get(sample_2)[1]
            target['difficulty'] = self.data_anno.get(sample_2)[2]
            target['completeness'] = (target['final_score'] / target['difficulty'])
            
            # print("[DEBUG-Train] sample_1 final_score:", data['final_score'])
            # print("[DEBUG-Train] sample_2 final_score:", target['final_score'])
            return data, target
        else:
            # test phrase
            if self.action_number_choosing:
                train_file_list = self.action_number_dict[self.data_anno[sample_1][0]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            elif self.DD_choosing:
                train_file_list = self.difficulties_dict[self.data_anno[sample_1][2]]
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            else:
                # randomly
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'], tmp['video_mask'], tmp['transits'], tmp['frame_labels'], tmp['pose_detections'] = self.load_video(item)
                # import pdb; pdb.set_trace()
                tmp['number'] = self.data_anno.get(item)[0]
                tmp['final_score'] = self.data_anno.get(item)[1]
                tmp['difficulty'] = self.data_anno.get(item)[2]
                tmp['completeness'] = (tmp['final_score'] / tmp['difficulty'])
                target_list.append(tmp)
                
            # print("[DEBUG-Test] sample_1 final_score:", data['final_score'])
            # for idx, target in enumerate(target_list):
                # print(f"[DEBUG-Test] exemplar {idx} final_score:", target['final_score'])
            return data, target_list

    def __len__(self):
        return len(self.dataset)


class DebugDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)