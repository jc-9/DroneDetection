"""
Justin Clay
ACD Drone Detection Demo
"""

from ultralytics import YOLO
import cv2
import numpy as np
import glob


def calibrate_images(camnum):
    chessboardSize = (7, 4)
    cap = cv2.VideoCapture(camnum)
    width = 1280  # Width in pixels
    height = 720  # Height in pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ret, frame = cap.read()
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        pass
    else:
        print('cap open')
    if not ret:
        print("Error: Failed to capture image.")
        pass
    h, w = frame.shape[:2]
    frameSize = (w, h)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    size_of_chessboard_squares_mm = 30
    objp = objp * size_of_chessboard_squares_mm

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('/Users/justinclay/PycharmProjects/AA_YoloV8/Calibration_Images/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 4), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs


def is_defined(var_name):
    try:
        eval(var_name)
        return True
    except NameError:
        return False


def crosshair(img, bbox, sub_line_length=10, sub_line_offset = 30):
    img_height, img_width = img.shape[:2]
    for box in bbox:
        # Extract YOLO format bounding box
        x_center, y_center, box_width, box_height = box.xywh[0]

        # Convert YOLO format to pixel coordinates
        x_center = int(x_center)
        y_center = int(y_center)


        # Calculate bounding box corners
        x_min = int(x_center - box_width // 2)
        x_max = int(x_center + box_width // 2)
        y_min = int(y_center - box_height // 2)
        y_max = int(y_center + box_height // 2)

        # Draw vertical line (crosshair)
        cv2.line(img, (x_center, y_min), (x_center, y_max), (0, 0, 255), 2)

        # Draw horizontal line (crosshair)
        cv2.line(img, (x_min, y_center), (x_max, y_center), (0, 0, 255), 2)

        # Vertical sub-crosshairs (closer to the center)
        cv2.line(img, (x_center - sub_line_length, y_center - sub_line_offset),
                 (x_center + sub_line_length, y_center - sub_line_offset), (0, 255, 0), 2)  # Above center
        cv2.line(img, (x_center - sub_line_length, y_center + sub_line_offset),
                 (x_center + sub_line_length, y_center + sub_line_offset), (0, 255, 0), 2)  # Below center

        # Horizontal sub-crosshairs (closer to the center)
        cv2.line(img, (x_center - sub_line_offset, y_center - sub_line_length),
                 (x_center - sub_line_offset, y_center + sub_line_length), (0, 255, 0), 2)  # Left of center
        cv2.line(img, (x_center + sub_line_offset, y_center - sub_line_length),
                 (x_center + sub_line_offset, y_center + sub_line_length), (0, 255, 0), 2)  # Right of center

    return img


def main(camnum, weights_path: str):
    global results
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(camnum)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        results = model(frame, conf=0.6, verbose=False)
        annotated_frame = crosshair(frame, results[0].boxes)
        cv2.imshow('Webcam Object Detection', annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cameraNumber = '/home/justin/kalvoai/drone_flying_test_video.m4v'
    weights_path = '/home/justin/PycharmProjects/DroneDetection/data/drone_test_results_09_30_242/weights/best.pt'
    main(cameraNumber, weights_path)