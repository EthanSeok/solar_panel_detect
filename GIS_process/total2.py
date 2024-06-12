import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from ultralytics import YOLO
import math
from pyproj import Transformer
import requests

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

def main(image_path, api_key):
    filename = image_path.split('/')[-1]
    x, y, z = get_tile_info_from_filename(filename)

    # 타일의 왼쪽 최하단 WTM 좌표 계산
    bottom_left_wtm_x, bottom_left_wtm_y = tile_to_wtm_bottom_left(x, y, z)
    print(f'Tile Bottom-Left WTM X: {bottom_left_wtm_x}, Y: {bottom_left_wtm_y}')

    # 타일의 크기 설정
    tile_size = 512

    # 특정 타일에서 1픽셀당 WTM 좌표 차이 계산
    wtm_per_pixel = calculate_wtm_per_pixel(tile_size, z)

    # YOLOv8 모델 로드 (사전 학습된 모델을 사용)
    model = YOLO('./models/best4.pt')

    # 이미지 로드
    image = cv2.imread(image_path)

    # YOLOv8 모델을 사용하여 객체 감지 수행
    results = model(image)

    # 이미지 복사본 생성 (annotated image)
    annotated_image = image.copy()

    addresses = []
    bbox_idx = 1
    # 바운딩 박스 중앙의 픽셀 번호 추출 및 위도/경도 계산
    for result in results:
        for box in result.boxes:
            # 바운딩 박스 좌표 (xmin, ymin, xmax, ymax)
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()

            # 중앙 좌표 계산
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            # 중앙 좌표 출력
            print(f'Bounding box center: ({center_x}, {center_y})')

            # 중심 픽셀의 WTM 좌표 계산
            center_wtm_x_offset = center_x * wtm_per_pixel
            center_wtm_y_offset = center_y * wtm_per_pixel

            # 중심 픽셀의 WTM 좌표
            pixel_wtm_x = bottom_left_wtm_x + center_wtm_x_offset
            pixel_wtm_y = bottom_left_wtm_y - center_wtm_y_offset

            # WTM 좌표를 WGS84 좌표로 변환
            pixel_lat, pixel_lon = wtm_to_wgs84(pixel_wtm_x, pixel_wtm_y)

            # 중앙 좌표에 원 표시 (시각화)
            cv2.circle(annotated_image, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            # 바운딩 박스 순서 표시 (테두리 흰색, 내부 녹색)
            label_x = int(xmin)  # xmin 좌표
            label_y = int(ymin) - 10  # ymin 좌표 위쪽에 위치

            # 번호가 이미지 바깥으로 나가지 않도록 조정
            label_y = max(label_y, 10)

            cv2.putText(annotated_image, str(bbox_idx), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(annotated_image, str(bbox_idx), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        2, cv2.LINE_AA)

            # 위도와 경도 출력
            print(f'Center Latitude: {pixel_lat}, Center Longitude: {pixel_lon}')

            # 위도와 경도로 주소 변환
            try:
                address, road_address = get_address(pixel_lat, pixel_lon, api_key)
                addresses.append((bbox_idx, address, road_address))
                print(f"{bbox_idx}. 지번 주소: {address}")
                print(f"{bbox_idx}. 도로명 주소: {road_address}")
                bbox_idx += 1
            except Exception as e:
                print(e)

    # BGR 이미지를 RGB 이미지로 변환 (Matplotlib는 RGB 형식을 사용)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # 타이틀에 주소 정보를 포함
    title = ''
    for idx, address, road_address in addresses:
        title += f'{idx}. 지번 주소: {address}, 도로명 주소: {road_address}\n'

    # Matplotlib를 사용하여 이미지 표시
    plt.figure(figsize=(10, 10), dpi=200)
    plt.imshow(annotated_image_rgb)
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    image_path = './output/jeonju_2/1911_2445_2.jpg'
    api_key = ""
    main(image_path, api_key)
