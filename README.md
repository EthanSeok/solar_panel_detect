## YOLO v8을 이용한 태양광 패널 검출

### 1. 프로젝트 개요
- 본 프로젝트는 YOLO v8을 이용하여 태양광 패널을 검출하는 것을 목표로 한다.
- 본 프로젝트는 태양광 패널의 설치 현황을 정량화하기 위한 기술로써 YOLO v8을 이용하여 태양광 패널을 검출하는 것을 목표로 한다.
- 이후 검출한 태양광 패널의 위치(위경도)를 이용하여 Geocoding 함으로써 주소지를 알아내고자 한다.
- 본 프로젝트를 통해 해당 주소지의 태양광 패널 설치 현황을 정량화하고자 한다.

### 2. 코드 실행
**코드실행은 p01번 부터 순차적으로 실행한다.**

- [p01_preprocess_label.py](p01_preprocess_label.py): 라벨 데이터를 전처리하는 코드이다. 처음 YOLO 학습을 위한 라벨 데이터가 matlab 프로젝트로 되어있었기 때문에, 
matlab 라벨 테이블을 파싱하여 파이썬에서 사용가능하도록 전처리하는 코드이다.


- [p02_coordinate2json.py](p02_coordinate2json.py): 바운딩 박스 전처리 및 Labelme json 포맷으로 변환한다. (json 내에 이미지 데이터가 인코딩되어 저장된다. --> roboflow 사용 편의성을 위해)


- [p03_data_augmentation.py](p03_data_augmentation.py): YOLO 모델의 정확도 향상을 위해 좌우반전, 상하반전 데이터 증강을 수행한다.


- [p03_data_augmentation_rotate.py](p03_data_augmentation_rotate.py): YOLO 모델의 정확도 향상을 위해 회전 데이터 증강을 수행한다.


- [p04_labelme2yolo.py](p04_labelme2yolo.py): Labelme json 포맷을 YOLO 학습을 위한 포맷으로 변환한다.


```
Anaconda Prompt:
pip install ultralytics ## yolo 설치
pip install wandb ## wandb 설치

conda activate yolo_env
cd 학습 데이터가 있는 폴더
wandb login
yolo detect train data=data.yaml model=yolov8n.pt epochs=300 imgsz=1024 batch=16 workers=0 ## yolo 학습 command
```

- [p05_yolo_detect.py](p05_yolo_detect.py): 학습된 YOLO 모델을 이용하여 테스트 해본다.


- [visualize_boundingbox.py](visualize_boundingbox.py): YOLO 모델을 이용하여 태양광 패널을 검출한 결과를 시각화해본다.


- [pt2onnx.py](pt2onnx.py): QGIS deepness plugin을 이용하여 YOLO 모델을 이용하여 태양광 패널을 검출하기 위해 onnx 포맷으로 weight 파일을 변환한다.


- [kakao_tile_download.py](GIS_process%2Fkakao_tile_download.py): 원하는 지역의 카카오 맵 타일을 다운로드 할 수 있다. [참고 링크](https://apis.map.kakao.com/web/sample/getTile/)


- [total.py](GIS_process%2Ftotal.py): YOLO 모델을 이용하여 태양광 패널을 검출한 결과를 시각화하고, 검출된 태양광 패널의 위치(위경도)를 이용하여 Geocoding을 수행한다.

## 결과

### 모델 결과
![모델 결과](https://github.com/EthanSeok/YOLO_v8_with_SAM/assets/93086581/2639e3d0-5cdc-4521-94c4-b963c676adde)


### 실제 지역
![실제 주소](https://github.com/EthanSeok/YOLO_v8_with_SAM/assets/93086581/736eaef9-98d8-4af0-b11b-71ec5d433651)
