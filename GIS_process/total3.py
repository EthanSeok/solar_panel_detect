import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from ultralytics import YOLO
import math
from pyproj import Transformer
import requests
import pandas as pd
from tqdm import tqdm

font_path = "C:/Windows/Fonts/KoPubDotumMedium.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False
rc('font', family=font)


def tile_to_wtm_bottom_left(x, y, z):
    wtm_x = x * math.pow(2, z - 3) * 256 - 30000
    wtm_y = (y + 1) * math.pow(2, z - 3) * 256 - 60000
    return wtm_x, wtm_y


def wtm_to_wgs84(wtm_x, wtm_y):
    transformer = Transformer.from_crs("EPSG:5181", "EPSG:4326")
    wgs84_lat, wgs84_lon = transformer.transform(wtm_y, wtm_x)  # Note the order change
    return wgs84_lat, wgs84_lon


def get_tile_info_from_filename(filename):
    parts = filename.split('_')
    x = int(parts[0])
    y = int(parts[1])
    z = int(parts[2].split('.')[0])
    return x, y, z


def calculate_wtm_per_pixel(tile_size, z):
    return (math.pow(2, z - 3) * 256) / tile_size


def get_address(lat, lng, api_key):
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"x": lng, "y": lat}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        result = response.json()
        if result['documents']:
            address_info = result['documents'][0]['address']
            road_address_info = result['documents'][0].get('road_address', None)
            address = address_info['address_name']
            road_address = road_address_info['address_name'] if road_address_info else "도로명 주소 없음"
            return address, road_address
        else:
            return "주소 정보 없음", "도로명 주소 없음"
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def process_image(image_path, api_key, model, result_dir):
    filename = image_path.split('/')[-1]
    x, y, z = get_tile_info_from_filename(filename)

    # 타일의 왼쪽 최하단 WTM 좌표 계산
    bottom_left_wtm_x, bottom_left_wtm_y = tile_to_wtm_bottom_left(x, y, z)
    # print(f'Tile Bottom-Left WTM X: {bottom_left_wtm_x}, Y: {bottom_left_wtm_y}')

    # 타일의 크기 설정
    tile_size = 512

    # 특정 타일에서 1픽셀당 WTM 좌표 차이 계산
    wtm_per_pixel = calculate_wtm_per_pixel(tile_size, z)

    # 이미지 로드
    image = cv2.imread(image_path)

    # YOLOv8 모델을 사용하여 객체 감지 수행
    results = model(image)

    addresses = []
    bbox_idx = 1
    annotated_image = image.copy()
    # 바운딩 박스 중앙의 픽셀 번호 추출 및 위도/경도 계산
    for result in results:
        for box in result.boxes:
            # 바운딩 박스 좌표 (xmin, ymin, xmax, ymax)
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()

            # 중앙 좌표 계산
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            # 중앙 좌표 출력
            # print(f'Bounding box center: ({center_x}, {center_y})')

            # 중심 픽셀의 WTM 좌표 계산
            center_wtm_x_offset = center_x * wtm_per_pixel
            center_wtm_y_offset = center_y * wtm_per_pixel

            # 중심 픽셀의 WTM 좌표
            pixel_wtm_x = bottom_left_wtm_x + center_wtm_x_offset
            pixel_wtm_y = bottom_left_wtm_y - center_wtm_y_offset

            # WTM 좌표를 WGS84 좌표로 변환
            pixel_lat, pixel_lon = wtm_to_wgs84(pixel_wtm_x, pixel_wtm_y)

            # 위도와 경도 출력
            # print(f'Center Latitude: {pixel_lat}, Center Longitude: {pixel_lon}')

            # 위도와 경도로 주소 변환
            try:
                address, road_address = get_address(pixel_lat, pixel_lon, api_key)
                addresses.append((bbox_idx, address, road_address))
                # print(f"{bbox_idx}. 지번 주소: {address}")
                # print(f"{bbox_idx}. 도로명 주소: {road_address}")
                bbox_idx += 1
            except Exception as e:
                print(e)

            # 바운딩 박스 그리기
            cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.circle(annotated_image, (center_x, center_y), 5, (0, 255, 0), -1)
            label_x = int(xmin)
            label_y = int(ymin) - 10
            label_y = max(label_y, 10)
            cv2.putText(annotated_image, str(bbox_idx), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(annotated_image, str(bbox_idx), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        2, cv2.LINE_AA)

    # 결과 이미지를 저장할 디렉토리가 없다면 생성
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 결과 이미지 파일 경로
    result_image_path = os.path.join(result_dir, filename)

    # 결과 이미지 저장
    cv2.imwrite(result_image_path, annotated_image)

    return addresses


def main(directory, api_key):
    model = YOLO('./models/best8.pt')
    results = []
    result_image_dir = './result_image/'

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            addresses = process_image(image_path, api_key, model, result_image_dir)
            for idx, address, road_address in addresses:
                results.append({
                    "filename": filename.rsplit('.', 1)[0],
                    "bbox_idx": idx,
                    "address": address,
                    "road_address": road_address
                })

    df = pd.DataFrame(results)
    df.to_csv('output_addresses.csv', index=False, encoding='utf-8-sig')
    print(df)


if __name__ == '__main__':
    directory = './output/jeonju_2/'
    api_key = "39b9c3a020b5438fe0c14225dde8a818"
    main(directory, api_key)
