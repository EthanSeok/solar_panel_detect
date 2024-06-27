import json
import os
import base64

def polygon_to_bbox(points):
    x_coordinates = points[0::2]
    y_coordinates = points[1::2]
    xmin = min(x_coordinates)
    xmax = max(x_coordinates)
    ymin = min(y_coordinates)
    ymax = max(y_coordinates)
    return [xmin, ymin, xmax, ymax]

def process_file(input_folder, image_folder, output_folder, file_name):
    input_file = os.path.join(input_folder, file_name + '.json')
    image_file = os.path.join(image_folder, file_name + '.jpg')

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Encode the image file in base64
    with open(image_file, 'rb') as img_f:
        img_data = img_f.read()
        img_data_base64 = base64.b64encode(img_data).decode('utf-8')

    # Prepare LabelMe format structure
    labelme_format = {
        "version": "4.5.6",  # Example version, use appropriate one
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_file),
        "imageData": img_data_base64,
        "imageHeight": 512,  # Fill in with actual image height if available
        "imageWidth": 512    # Fill in with actual image width if available
    }

    # Convert each polygon to bounding box and add to LabelMe format
    for obj in data["Learning Data Info"]["objects"]:
        if obj["annotation_type"] == "polygon":
            bbox = polygon_to_bbox(obj["points"])
            shape = {
                # "label": obj["class_name"],
                "label": 'solar panel',
                "points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            labelme_format["shapes"].append(shape)

    # Save the new JSON in LabelMe format
    output_file = os.path.join(output_folder, file_name + ".json")
    with open(output_file, 'w') as f:
        json.dump(labelme_format, f, indent=4)

    print(f"LabelMe formatted JSON saved to: {output_file}")

def main(input_folder, image_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            base_name = os.path.splitext(file_name)[0]
            process_file(input_folder, image_folder, output_folder, base_name)

if __name__ == "__main__":
    input_folder = r'D:\234.태양광 발전 현황 및 적지 분석 데이터\01-1.정식개방데이터\Training\02.라벨링데이터\TL_TL_01.라벨링데이터_태양광 발전현황 데이터_25cm급 항공이미지_Solar panel'
    image_folder = r'D:\234.태양광 발전 현황 및 적지 분석 데이터\01-1.정식개방데이터\Training\01.원천데이터\preprocessed'
    output_folder = 'output/labelme_label'
    os.makedirs(output_folder, exist_ok=True)
    main(input_folder, image_folder, output_folder)
