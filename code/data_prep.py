import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def convert_to_yolo_format(data_dir, annotations_file):
    """
    Converts faces.csv to YOLO format (.txt files).
    CSV format: image_name, width, height, x0, y0, x1, y1
    """
    labels_dir = os.path.join(data_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    df = pd.read_csv(annotations_file)

    for _, row in df.iterrows():
        img_name = row['image_name']


        img_w, img_h = row['width'], row['height']
        x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']


        x_center = ((x0 + x1) / 2) / img_w
        y_center = ((y0 + y1) / 2) / img_h
        width = (x1 - x0) / img_w
        height = (y1 - y0) / img_h


        label_file = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(label_file, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print("Labels created in:", labels_dir)


def create_yolo_dataset(data_dir, output_dir):
    """
    Creates the YOLO dataset structure with train/val splits.
    """
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')


    image_list = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]


    train_images, val_images = train_test_split(image_list, test_size=0.2, random_state=42)


    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)


    for img_name in train_images:
        shutil.copy(os.path.join(images_dir, img_name), os.path.join(output_dir, 'images', 'train'))
        shutil.copy(os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt'),
                    os.path.join(output_dir, 'labels', 'train'))

    for img_name in val_images:
        shutil.copy(os.path.join(images_dir, img_name), os.path.join(output_dir, 'images', 'val'))
        shutil.copy(os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt'),
                    os.path.join(output_dir, 'labels', 'val'))

    print("Dataset created in:", output_dir)


if __name__ == "__main__":
    raw_data_path = r"L:\Humanface_detection\raw_data"
    yolo_data_path = r"L:\Humanface_detection\yolo_dataset"
    annotations_csv = os.path.join(raw_data_path, "faces.csv")

    print("Step 1: Converting annotations to YOLO format...")
    convert_to_yolo_format(raw_data_path, annotations_csv)

    print("Step 2: Creating YOLO dataset structure...")
    create_yolo_dataset(raw_data_path, yolo_data_path)

    print("Dataset preparation complete.")

yaml_file = os.path.join(os.path.dirname(yolo_data_path), "faces_data.yaml")
with open(yaml_file, "w") as f:
    f.write(f"train: {yolo_data_path}/images/train\n")
    f.write(f"val: {yolo_data_path}/images/val\n")
    f.write("nc: 1\n")
    f.write("names: ['face']\n")

print("faces_data.yaml created at:", yaml_file)

