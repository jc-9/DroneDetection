"""
Justin Clay
Dataset Adudit tool - Iterates though the yolo data set and displays the bounding boxes.
"""

import os
import cv2
import numpy as np
from fontTools.varLib.errors import FoundANone
from ultralytics.data.augment import Albumentations
import albumentations as A

BOX_COLOR = (0, 255, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    height, width, _ = img.shape

    """Visualizes a single bounding box on the image"""
    center_x, center_y, box_width, box_height = bbox

    x1 = int((center_x - box_width / 2) * width)
    y1 = int((center_y - box_height / 2) * height)
    x2 = int((center_x + box_width / 2) * width)
    y2 = int((center_y + box_height / 2) * height)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    label = f"Class {class_name}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def load_yolo_annotations(annotation_file):
    """
    Load YOLO annotations from a text file.
    Returns a list of bounding boxes and class ids.
    Each bounding box is in the format (class_id, center_x, center_y, width, height)
    """
    boxes = []
    with open(annotation_file, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            boxes.append((class_id, center_x, center_y, width, height))
    print(file.name)
    print(boxes)
    return boxes


def draw_bounding_boxes(image, boxes):
    """
    Draw bounding boxes on the image.
    """
    height, width, _ = image.shape
    for box in boxes:
        class_id, center_x, center_y, box_width, box_height = box
        x1 = int((center_x - box_width / 2) * width)
        y1 = int((center_y - box_height / 2) * height)
        x2 = int((center_x + box_width / 2) * width)
        y2 = int((center_y + box_height / 2) * height)

        # Draw rectangle and label
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"Class {class_id}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def transform(img, boxes):
    class_labels = [i[0] for i in boxes]
    bbox = [list(i[1:5]) for i in boxes]

    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
        # A.Blur(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['label_fields'])
    )

    img_transformed_dict = transform(image=img, bboxes=bbox, label_fields=class_labels)
    img = img_transformed_dict['image']
    boxes = img_transformed_dict['bboxes']
    id = img_transformed_dict['label_fields']
    return img, boxes, id

def yolo_to_crosshair(img, bbox, sub_line_length=10):
    """
    Convert YOLO bounding box format to a crosshair shape on the image.

    Parameters:
    img: The input image (numpy array).
    bbox: The bounding box in YOLO format [x_center, y_center, width, height] (relative to image size).
    """
    img_height, img_width = img.shape[:2]
    for box in bbox:
        # Extract YOLO format bounding box
        _,x_center_rel, y_center_rel, width_rel, height_rel = box

        # Convert YOLO format to pixel coordinates
        x_center = int(x_center_rel * img_width)
        y_center = int(y_center_rel * img_height)
        box_width = int(width_rel * img_width)
        box_height = int(height_rel * img_height)

        # Calculate bounding box corners
        x_min = x_center - box_width // 2
        x_max = x_center + box_width // 2
        y_min = y_center - box_height // 2
        y_max = y_center + box_height // 2

        # Draw vertical line (crosshair)
        cv2.line(img, (x_center, y_min), (x_center, y_max), (0, 0, 255), 2)

        # Draw horizontal line (crosshair)
        cv2.line(img, (x_min, y_center), (x_max, y_center), (0, 0, 255), 2)
        # Vertical sub-crosshairs
        cv2.line(img, (x_center - sub_line_length, y_min), (x_center + sub_line_length, y_min), (0, 255, 0), 2)  # Top
        cv2.line(img, (x_center - sub_line_length, y_max), (x_center + sub_line_length, y_max), (0, 255, 0),
                 2)  # Bottom

        # Horizontal sub-crosshairs
        cv2.line(img, (x_min, y_center - sub_line_length), (x_min, y_center + sub_line_length), (0, 255, 0), 2)  # Left
        cv2.line(img, (x_max, y_center - sub_line_length), (x_max, y_center + sub_line_length), (0, 255, 0), 2)  # Right

    # return img

def display_images_with_bboxes(image_folder, annotation_folder):
    global boxes
    """
    Display images with bounding boxes from YOLO dataset.
    """
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(('.jpg', '.jpeg', '.PNG', '.bmp', '.JPEG')):
            image_path = os.path.join(image_folder, image_filename)
            annotation_path = os.path.join(annotation_folder, os.path.splitext(image_filename)[0] + '.txt')

            # Read image
            img = cv2.imread(image_path)
            if img is None:
                continue

            # Load annotations
            if os.path.exists(annotation_path):
                boxes = load_yolo_annotations(annotation_path)

            # draw_bounding_boxes(img, boxes)
            yolo_to_crosshair(img,boxes)
            transform_flag = False
            if transform_flag:
                img_transformed, boxes, classid = transform(img, boxes)
                for bbox, classid in zip(boxes, classid):
                    img_transformed = visualize_bbox(img_transformed, bbox, classid)

            # Display image
            cv2.imshow('Normal Image', img)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            if transform_flag:
                cv2.imshow('transformed Image', img_transformed)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
                cv2.destroyAllWindows()


# Define the paths to your dataset
image_folder = '/home/justin/PycharmProjects/DroneDetection/data/ds1/dataset_txt/images/train'
annotation_folder = '/home/justin/PycharmProjects/DroneDetection/data/ds1/dataset_txt/labels/train'

# Run the script
display_images_with_bboxes(image_folder, annotation_folder)
