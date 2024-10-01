import cv2

def yolo_to_crosshair(img, bbox):
    """
    Convert YOLO bounding box format to a crosshair shape on the image.

    Parameters:
    img: The input image (numpy array).
    bbox: The bounding box in YOLO format [x_center, y_center, width, height] (relative to image size).
    """
    img_height, img_width = img.shape[:2]

    # Extract YOLO format bounding box
    x_center_rel, y_center_rel, width_rel, height_rel = bbox

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
    cv2.line(img, (x_center, y_min), (x_center, y_max), (0, 255, 0), 2)

    # Draw horizontal line (crosshair)
    cv2.line(img, (x_min, y_center), (x_max, y_center), (0, 255, 0), 2)

    return img

img_path = '/home/justin/PycharmProjects/DroneDetection/data/ds1/dataset_txt/images/train/0001.jpg'
txt_file = '/home/justin/PycharmProjects/DroneDetection/data/ds1/dataset_txt/labels/train/0001.txt'