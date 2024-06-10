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

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def image_to_base64(image_array):
    image_pil = Image.fromarray(np.uint8(image_array * 255))
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def warpAffine(src, M, dsize, from_bounding_box_only=False):
    return cv2.warpAffine(src, M, dsize)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    image = warpAffine(image, M, (nW, nH), False)

    return image

def crop_to_center(old_img, new_img):
    if isinstance(old_img, tuple):
        original_shape = old_img
    else:
        original_shape = old_img.shape
    original_width = original_shape[1]
    original_height = original_shape[0]
    original_center_x = original_shape[1] / 2
    original_center_y = original_shape[0] / 2

    new_width = new_img.shape[1]
    new_height = new_img.shape[0]
    new_center_x = new_img.shape[1] / 2
    new_center_y = new_img.shape[0] / 2

    new_left_x = int(max(new_center_x - original_width / 2, 0))
    new_right_x = int(min(new_center_x + original_width / 2, new_width))
    new_top_y = int(max(new_center_y - original_height / 2, 0))
    new_bottom_y = int(min(new_center_y + original_height / 2, new_height))

    canvas = np.zeros(original_shape)

    left_x = int(max(original_center_x - new_width / 2, 0))
    right_x = int(min(original_center_x + new_width / 2, original_width))
    top_y = int(max(original_center_y - new_height / 2, 0))
    bottom_y = int(min(original_center_y + new_height / 2, original_height))

    canvas[top_y:bottom_y, left_x:right_x] = new_img[new_top_y:new_bottom_y, new_left_x:new_right_x]

    return canvas


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def trim_point(c1, x0, y0, width, height):
    return (min(max(0, x0+c1[0]), width),
            min(max(0, y0+c1[1]), height))


def rotate_annotation(origin, annotation, degree):
    new_annotation = copy.deepcopy(annotation)

    angle = math.radians(degree)
    origin_x, origin_y = origin
    origin_y *= -1

    x = annotation["x"]
    y = annotation["y"]

    new_x, new_y = map(lambda x: round(x * 2) / 2, rotate_point(
        (origin_x, origin_y), (x, -y), angle)
                       )

    new_annotation["x"] = new_x
    new_annotation["y"] = -new_y

    width = annotation["width"]
    height = annotation["height"]

    left_x = x - width / 2
    right_x = x + width / 2
    top_y = y - height / 2
    bottom_y = y + height / 2

    c1 = (left_x, top_y)
    c2 = (right_x, top_y)
    c3 = (right_x, bottom_y)
    c4 = (left_x, bottom_y)

    c1 = rotate_point(origin, c1, angle)
    c2 = rotate_point(origin, c2, angle)
    c3 = rotate_point(origin, c3, angle)
    c4 = rotate_point(origin, c4, angle)

    x_coords, y_coords = zip(c1, c2, c3, c4)
    new_annotation["width"] = round(max(x_coords) - min(x_coords))
    # print(c1, c2, c3, c4)
    # print('x_coords', x_coords)
    # print('y_coords', y_coords)

    new_annotation["height"] = round(max(y_coords) - min(y_coords))
    # print('x', new_x, min(x_coords))
    # print('y', -new_y, min(y_coords))
    # new_annotation["x"] = min(x_coords)
    # new_annotation["y"] = min(y_coords)

    # print("before", c1, c2, c3, c4)
    # cn1 = trim_point(c1, new_x, -new_y, origin_x*2, origin_y*-2)
    # cn2 = trim_point(c2, new_x, -new_y, origin_x*2, origin_y*-2)
    # cn3 = trim_point(c3, new_x, -new_y, origin_x*2, origin_y*-2)
    # cn4 = trim_point(c4, new_x, -new_y, origin_x*2, origin_y*-2)
    # # print("after trim", c1, c2, c3, c4)
    #
    # x_coords, y_coords = zip(cn1, cn2, cn3, cn4)
    # new_annotation["width"] = round(max(x_coords) - min(x_coords))
    # new_annotation["height"] = round(max(y_coords) - min(y_coords))
    # new_annotation["x"] = min(x_coords)
    # new_annotation["y"] = min(y_coords)

    x0 = new_annotation["x"] - new_annotation["width"]/2
    y0 = new_annotation["y"] - new_annotation["height"]/2
    x1 = x0 + new_annotation["width"]
    y1 = y0 + new_annotation["height"]
    # print(x0, x1, y0, y1)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(origin_x*2, x1)
    y1 = min(-origin_y*2, y1)
    new_annotation["x"] = (x0 + x1)/2
    new_annotation["y"] = (y0 + y1)/2
    new_annotation["width"] = x1 - x0
    new_annotation["height"] = y1 - y0
    # print('after', x0, x1, y0, y1)

    return new_annotation

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

def save_to_labelme_json(filepath, image_base64, annotations, image_shape):
    """Save annotations and image data to a labelme JSON file."""
    data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": "image.png",
        "imageData": image_base64,
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[1]
    }

    for annot in annotations:
        shape = {
            "label": annot["label"],
            "points": [
                [annot["x"] - annot["width"] / 2, annot["y"] - annot["height"] / 2],
                [annot["x"] + annot["width"] / 2, annot["y"] + annot["height"] / 2]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        data["shapes"].append(shape)

    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, indent=2)

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

    # plt.axis('off')
    plt.show()

def main():
    directory = "./output/labelme_label/"
    output_image_dir = "./output/augmentation/images/"
    output_label_dir = "./output/augmentation/label/"

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    filenames = [filename for filename in os.listdir(directory) if filename.endswith(".json")]
    print(filenames)

    for json_data in filenames:
        with open(os.path.join(directory, json_data), 'r', encoding='utf-8') as file:
            data = json.load(file)

        im = np.array(read_image(data)).astype(np.float64) / 255

        for angle in range(10, 360, 10):
            rotated = rotate_image(im, angle)
            cropped = crop_to_center(im, rotated)
            cropped_image = Image.fromarray((cropped * 255).astype(np.uint8))

            image_filename = f'rotated_{angle:03d}_' + json_data.split(".")[0] + '.png'
            cropped_image_path = os.path.join(output_image_dir, image_filename)
            cropped_image.save(cropped_image_path)

            annotations = calculate_annotations(data)
            rotated_annotations = [rotate_annotation((im.shape[1] / 2, im.shape[0] / 2), annot, angle) for annot in
                                   annotations]

            image_base64 = image_to_base64(cropped)
            json_filename = f'rotated_{angle:03d}_' + json_data.split(".")[0] + '.json'
            json_file_path = os.path.join(output_label_dir, json_filename)
            save_to_labelme_json(json_file_path, image_base64, rotated_annotations, cropped.shape)

            # visualize_annotations(cropped, rotated_annotations)

if __name__ == '__main__':
    main()

