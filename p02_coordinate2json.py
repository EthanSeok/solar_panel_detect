import pandas as pd
import json
import os
import base64
from PIL import Image

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  

def convert_coordinates(x, y, w, h):
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]

def save_labelme_json(image_path, coordinates, output_dir):
    image_filename = os.path.basename(image_path)
    image_data = encode_image_to_base64(image_path)
    image_width, image_height = get_image_size(image_path)
    shapes = [
        {
            "label": "solar pannel",
            "points": [
                [coords[0], coords[1]],
                [coords[2], coords[3]]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        for coords in coordinates
    ]
    json_data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    json_filename = os.path.splitext(image_filename)[0] + '.json'
    with open(os.path.join(output_dir, json_filename), 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def main():
    file_path = './output/preprocess_label/preprocessed_label.csv'
    output_directory = './output/labelme_label'

    data = pd.read_csv(file_path)
    os.makedirs(output_directory, exist_ok=True)
    for idx, row in data.iterrows():
        coordinates_raw = eval(row['coordinate'])
        coordinates = [convert_coordinates(x, y, w, h) for x, y, w, h in coordinates_raw]
        image_directory = row['files']
        save_labelme_json(image_directory, coordinates, output_directory)

if __name__ == "__main__":
    main()
