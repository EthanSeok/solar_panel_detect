import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from ultralytics import YOLO
import math
from pyproj import Proj, Transformer
import requests

font_path = "C:/Windows/Fonts/KoPubDotumMedium.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False
rc('font', family=font)

def tile_to_wtm(x, y, z):
    wtm_x = x * math.pow(2, z - 3) * 256 - 30000
    wtm_y = y * math.pow(2, z - 3) * 256 - 60000
    return wtm_x, wtm_y

def wtm_to_wgs84(x, y, z):
    transformer = Transformer.from_crs("EPSG:5174", "EPSG:4326")
    wtm_x, wtm_y = tile_to_wtm(x, y, z)
    wgs84_lat, wgs84_lon = transformer.transform(wtm_y, wtm_x)  # Note the order change
    return wgs84_lat, wgs84_lon

def get_tile_info_from_filename(filename):
    parts = filename.split('_')
    x = int(parts[1])
    y = int(parts[0])
    z = int(parts[2].split('.')[0])
    return x, y, z

def calculate_lon_per_pixel(lat, tile_size, lon_diff):
    meters_per_degree_lon = math.cos(math.radians(lat)) * 111320  # 1도 경도의 거리 (미터)
    lon_per_pixel = lon_diff / tile_size
    return lon_per_pixel / meters_per_degree_lon

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

    # 타일의 중앙 WGS84 위도 경도 계산
    center_lat, center_lon = wtm_to_wgs84(x, y, z)
    print(f'Tile Center Latitude: {center_lat}, Longitude: {center_lon}')

    # 타일의 크기와 경도 차이 설정
    tile_size = 512
    lon_diff = 0.002783717

    # 특정 위도에서 1픽셀당 경도 차이 계산
    lon_per_pixel = calculate_lon_per_pixel(center_lat, tile_size, lon_diff)

    # YOLOv8 모델 로드 (사전 학습된 모델을 사용)
    model = YOLO('./models/best.pt')

    # 이미지 로드
    image = cv2.imread(image_path)

    # YOLOv8 모델을 사용하여 객체 감지 수행
    results = model(image)

    # 이미지 복사본 생성 (annotated image)
    annotated_image = image.copy()

    addresses = []
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

            # 중심 픽셀의 경도 계산
            center_lon_offset = (center_x - (tile_size / 2)) * lon_per_pixel
            center_lat_offset = (center_y - (tile_size / 2)) * lon_per_pixel  # 단순히 경도로 변환 (위도 계산 필요)

            # 중심 픽셀의 위도, 경도 계산
            pixel_lon = center_lon + center_lon_offset
            pixel_lat = center_lat - center_lat_offset

            # 중앙 좌표에 원 표시 (시각화)
            cv2.circle(annotated_image, (center_x, center_y), 5, (0, 255, 0), -1)

            # 위도와 경도 출력
            print(f'Center Latitude: {pixel_lat}, Center Longitude: {pixel_lon}')

            # 위도와 경도로 주소 변환
            try:
                address, road_address = get_address(pixel_lat, pixel_lon, api_key)
                addresses.append((address, road_address))
                print(f"지번 주소: {address}")
                print(f"도로명 주소: {road_address}")
            except Exception as e:
                print(e)

    # BGR 이미지를 RGB 이미지로 변환 (Matplotlib는 RGB 형식을 사용)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # 타이틀에 주소 정보를 포함
    title = ''
    for i, (address, road_address) in enumerate(addresses):
        title += f'{y}_{x}_{z}: {address}\n'

    # Matplotlib를 사용하여 이미지 표시
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image_rgb)
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    image_path = './output/757_774_3.jpg'
    api_key = ""
    main(image_path, api_key)
