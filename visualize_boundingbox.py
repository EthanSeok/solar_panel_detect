import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import math
import copy
import json
import base64
from io import BytesIO
import io
import cv2
import os

def read_image(data):
    base64_image_data = data['imageData']
    image_data = base64.b64decode(base64_image_data)

    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image

def calculate_annotations(data):
    annotations = []
    for shape in data['shapes']:
        xmin = shape['points'][0][0]
        ymin = shape['points'][0][1]
        xmax = shape['points'][1][0]
        ymax = shape['points'][1][1]

        width = xmax - xmin
        height = ymax - ymin

        annotations.append({
            "label": "solar pannel",
            "x": (xmin + xmax) / 2,
            "y": (ymin + ymax) / 2,
            "width": width,
            "height": height
        })
    return annotations

def visualize_annotations(image, annotations):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap='gray')

    for annot in annotations:
        x_center = annot['x']
        y_center = annot['y']
        width = annot['width']
        height = annot['height']

        rect = patches.Rectangle((x_center - width / 2, y_center - height / 2),
                                 width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        label_y_position = y_center - height / 2 - 10  # 텍스트를 위로 10 픽셀 올립니다.
        plt.text(x_center, label_y_position, annot['label'], color='white', fontsize=12,
                 ha='center', va='bottom', bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

def main():
    ## json 경로
    directory = "./output/augmentation/label"

    ## Labelme JSON 파일명
    json_data = 'rotated_350_w329.json'

    with open(os.path.join(directory, json_data), 'r', encoding='utf-8') as file:
        data = json.load(file)

    im = np.array(read_image(data)).astype(np.float64) / 255
    annotations = calculate_annotations(data)
    visualize_annotations(im, annotations)


if __name__ == '__main__':
    main()