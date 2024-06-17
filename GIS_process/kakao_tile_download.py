import requests
import os
import time


def download_image(zoom, xtile, ytile, output_dir):
    url = f"https://map0.daumcdn.net/map_skyview_hd/L{zoom}/{ytile}/{xtile}.jpg"
    file_path = os.path.join(output_dir, f"{xtile}_{ytile}_{zoom}.jpg")

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"파일이 성공적으로 {file_path}에 다운로드 되었습니다.")
    else:
        print(f"파일 다운로드 실패. 상태 코드: {response.status_code} - {file_path}")


def main():
    zoom = 2
    output_dir = "output/jeonju_2/"

    # output 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0

    for xtile in range(1856, 1921): ## xtile == 첫 번째 숫자
        for ytile in range(2432, 2446): ## ytile == 두 번째 숫자
            download_image(zoom, xtile, ytile, output_dir)
            count += 1

            # 100장을 다운로드한 후 5초간 휴식
            if count % 100 == 0:
                print("100장을 다운로드했습니다. 5초간 휴식합니다...")
                time.sleep(5)


if __name__ == '__main__':
    main()
