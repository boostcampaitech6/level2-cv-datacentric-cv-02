# [AI Tech 6th CV Course] 팀 도라에몽

## Table of content

- [Overview](#Overview)
- [Member](#Member)
- [Approach](#Approach)
- [Result](#Result)
- [File Tree](#filetree)
- [Usage](#Code)




<br></br>
## Overview <a id = 'Overview'></a>
![image1](https://github.com/kylew1004/Python_projects/assets/5775698/03e971cd-56e0-42c2-ba3f-d9ecccaa272a)
OCR(Optimal Character Recognition) 기술 중 글자 검출(text detection)을 수행합니다. 
베이스라인 모델인 EAST 모델을 수정하지 않고 Data Centric을 통해 성능을 올리는 것이 프로젝트 목표입니다.
<br></br>

## Member <a id = 'Member'></a>

|김찬우|남현우|류경엽|이규섭|이현지|한주희|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<a href='https://github.com/uowol'><img src='https://avatars.githubusercontent.com/u/20416616?v=4' width='100px'/></a>|<a href='https://github.com/nhw2417'><img src='https://avatars.githubusercontent.com/u/103584775?s=88&v=4' width='100px'/></a>|<a href='https://github.com/kylew1004'><img src='https://avatars.githubusercontent.com/u/5775698?s=88&v=4' width='100px'/></a>|<a href='https://github.com/9sub'><img src='https://avatars.githubusercontent.com/u/113101019?s=88&v=4' width='100px'/></a>|<a href='https://github.com/solee328'><img src='https://avatars.githubusercontent.com/u/22787039?s=88&v=4' width='100px'/></a>|<a href='https://github.com/jh7316'><img src='https://avatars.githubusercontent.com/u/95545960?s=88&v=4' width='100px'/></a>|


<br></br>

## Approach <a id = 'Approach'></a>
### Architecture
- EAST
### Dataset
- medical 제공 데이터
- [공공데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=88)

<br></br>

## Result <a id = 'Result'></a>

### Scoreboard
<img width="1066" alt="image" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-02/assets/5775698/2db6bbc6-ae23-434f-ac2a-47ac902e9237">

### Wrap Reports
<a href="" target="_blank">Data Centric Wrapup Reports</a>


<br></br>

## File Tree <a id = 'filetree'></a>
```
level2-objectdetection-cv-02
|
.
.
|
|──ensemble
|   |── ensemble.ipynb
|   └── wbf_ensemble.py
|── convert_ann_format.py
|── dataset.py
|── detect.py
|── deteval.py
|── east_dataset.py
|── inference.py
|── loss.py
|── merge_annotations.py
|── model.py
|── train.py
└── README.md
```

<br></br>

## Usage <a id = 'Code'></a>

### Package install

```bash
pip install -r requirements.txt
```

### Model Train & Inference
```bash
python train.py

python inference.py
```


### Extra Tools
1. Convert Annotation Format, UFO annotation format과 cvat에서 지원하는 datumaro annotation format간의 변환을 실행합니다.
2. Merge Annotations, 두 개의 UFO annotation 파일을 합쳐 하나의 파일로 반환합니다.
```py
# merge_annotations.py 
source_dir = '../data/medical/ufo/train.json'   # 원본 데이터의 경로를 입력합니다.
target_dir = '../aug/merged.json'               # 병합하고자 하는 데이터의 경로를 입력합니다.
output_dir = '../data/medical/ufo/merged.json'  # 병합된 파일이 저장될 경로를 입력합니다    
start_idx = 0   # 
num_iter = 120
```
3. Make Custom Noise, 프린터기에서 발생하는 노이즈와 유사한 노이즈를 발생시킬 수 있습니다.

### [WBF ensemble](./docs/wbf_ensemble.md)
1. WBF ensemble 특성상 bbox 형태가 polygon이 아닌 사각형 형태여야 하기 때문에 coco format으로 변환된 test.json을 생성합니다.
```py
# test dataset이 있는 폴더를 입력합니다
test_folder = '/data/ephemeral/home/data/medical/img/test/'
```
2. 앙상블 하고자 하는 파일을 업로드합니다.
```py
# 앙상블 하고싶은 파일을 모아둔 폴더를 입력합니다
for file_name in os.listdir('/data/ephemeral/home/data/medical/ensemble/json'):
```
3. coco format으로 저장된 test.json파일의 경로를 입력합니다.
```py
annotation = '/data/ephemeral/home/data/medical/output_ensemble/test.json'
```
4.  iou_thr, skip_box_thr, weight 를 조정해 WBF ensemble을 진행합니다. 
