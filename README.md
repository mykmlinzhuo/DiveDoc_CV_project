# 🏊 DiveDoc: Structured Diagnosis of Diving Performance via Pose-Guided Action Parsing
DiveDoc is a fine-grained action quality assessment (AQA) model for diving videos. It parses human-centric motion into spatial-temporal steps using pose guidance and performs structured comparisons with exemplars to generate interpretable scores.
[paper](paper/DiveDoc.pdf)


## Requirements

Make sure the following dependencies installed (python):

* pytorch >= 0.4.0
* matplotlib=3.1.0
* einops
* timm
* torch_videovision

```
pip install git+https://github.com/hassony2/torch_videovision
```

## Dataset & Annotations

### FineDiving Download

To download the FineDiving dataset and annotations, please follow [FineDiving](https://github.com/xujinglin/FineDiving).

### FineDiving-HM Download
To download FineDiving-HM, please sign the [Release Agreement](agreement/Release_Agreement.pdf) and send it to send it to Jinglin Xu (xujinglinlove@gmail.com). By sending the application, you are agreeing and acknowledging that you have read and understand the notice. We will reply with the file and the corresponding guidelines right after we receive your request!


The format of FineDiving-HM is consistent with FineDiving. Please place the downloaded FineDiving-HM in `data`.

You could also seek to any of the two authors for the dataset and annotations.

### Data Structure

```
$DATASET_ROOT
├── FineDiving
|  ├── FINADivingWorldCup2021_Men3m_final_r1
|     ├── 0
|        ├── 00489.jpg
|        ...
|        └── 00592.jpg
|     ...
|     └── 11
|        ├── 14425.jpg
|        ...
|        └── 14542.jpg
|  ...
|  └── FullMenSynchronised10mPlatform_Tokyo2020Replays_2
|     ├── 0
|     ...
|     └── 16 
└──FineDiving_HM
|  ├── FINADivingWorldCup2021_Men3m_final_r1
|     ├── 0
|        ├── 00489.jpg
|        ...
|        └── 00592.jpg
|     ...
|     └── 11
|        ├── 14425.jpg
|        ...
|        └── 14542.jpg
|  ...
|  └── FullMenSynchronised10mPlatform_Tokyo2020Replays_2
|     ├── 0
|     ...
|     └── 16 

$ANNOTATIONS_ROOT
|  ├── FineDiving_coarse_annotation.pkl
|  ├── FineDiving_fine-grained_annotation.pkl
|  ├── Sub_action_Types_Table.pkl
|  ├── fine-grained_annotation_aqa.pkl
|  ├── train_split_5.pkl
|  ├── test_split_5.pkl
```

## Training
Training on 1*NVIDIA A100 40G GPU.

To download pretrained_i3d_wight, please follow [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/tree/master), and put `model_rgb.pth` in `models` folder.

To train the model, please run:
```bash
python launch_single.py
```

## Test
To test the trained model, please set `test: True` in [config](FineDiving_FineParser.yaml) and run:
```bash
python launch_single.py
```

