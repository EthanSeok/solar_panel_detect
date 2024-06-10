import os
import json
from PIL import Image
import base64
from io import BytesIO

def read_image(data):
    base64_image_data = data['imageData']
    image_data = base64.b64decode(base64_image_data)

    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def flip_image(image, mode='horizontal'):
    if mode == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)

def flip_coordinates(coords, image_width, image_height, mode='horizontal'):
    if mode == 'horizontal':
        return [[image_width - x, y] for [x, y] in coords]
    elif mode == 'vertical':
        return [[image_width - x, image_height - y] for [x, y] in coords]

def save_flipped_image(image, output_path, mode, filename):
    image.save(os.path.join(output_path, f'{mode}_{filename}'))

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images_output_folder = os.path.join(output_folder, 'images')
    labels_output_folder = os.path.join(output_folder, 'label')
    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(labels_output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            with open(os.path.join(input_folder, filename), 'r') as f:
                label_data = json.load(f)

            image = read_image(label_data)

            # Process for horizontal flip
            flipped_image = flip_image(image, mode='horizontal')
            flipped_filename = filename.replace('.json', '.png')
            save_flipped_image(flipped_image, images_output_folder, 'flipped', flipped_filename)
            flipped_image_base64 = encode_image_to_base64(flipped_image)
            flipped_label_data = label_data.copy()
            flipped_label_data["imagePath"] = flipped_filename
            flipped_label_data["imageData"] = flipped_image_base64
            for shape in flipped_label_data["shapes"]:
                shape["points"] = flip_coordinates(shape["points"], flipped_image.width, flipped_image.height)
            flipped_label_path = os.path.join(labels_output_folder, f'flipped_{filename}')
            with open(flipped_label_path, 'w') as f:
                json.dump(flipped_label_data, f, indent=4)

            # Process for vertical flip
            vertically_flipped_image = flip_image(image, mode='vertical')
            vertically_flipped_filename = filename.replace('.json', '.png')
            save_flipped_image(vertically_flipped_image, images_output_folder, 'vertically_flipped', vertically_flipped_filename)
            vertically_flipped_image_base64 = encode_image_to_base64(vertically_flipped_image)
            vertically_flipped_label_data = label_data.copy()
            vertically_flipped_label_data["imagePath"] = vertically_flipped_filename
            vertically_flipped_label_data["imageData"] = vertically_flipped_image_base64
            for shape in vertically_flipped_label_data["shapes"]:
                shape["points"] = flip_coordinates(shape["points"], vertically_flipped_image.width, vertically_flipped_image.height, mode='vertical')
            vertically_flipped_label_path = os.path.join(labels_output_folder, f'vertically_flipped_{filename}')
            with open(vertically_flipped_label_path, 'w') as f:
                json.dump(vertically_flipped_label_data, f, indent=4)

if __name__ == "__main__":
    input_folder = './output/labelme_label/'
    output_folder = './output/augmentation/'
    process_images(input_folder, output_folder)
