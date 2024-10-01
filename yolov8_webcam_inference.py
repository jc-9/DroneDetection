"""
Justin Clay
Run Yolo inference with local Webcam stream using traditional webcam and not a VPU Webcam (i.e. OAK-D, Luxonis)
"""

from ultralytics import YOLO
import cv2
import os.path
import numpy as np
import glob
import torch
import datetime


# def capture_image(save_path):
#     cap = cv2.VideoCapture(0)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
#     img_number = 0
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#
#         if not ret:
#             print("Error: Failed to capture image.")
#             break
#
#         # Display the resulting frame
#         cv2.imshow('Press "c" to capture image, "q" to quit', frame)
#
#         # Wait for a key press
#         key = cv2.waitKey(1) & 0xFF
#
#         if key == ord('c'):  # 'c' to capture the image
#             # Save the captured image
#             filename = f'calibrationImage_{img_number}.jpg'
#             path = os.path.join(save_path, filename)
#             img_number += 1
#             cv2.imwrite(path, frame)
#             print(f"Image saved to {save_path}")
#             # break
#         elif key == ord('q'):  # 'q' to quit without capturing
#             print("Quit without capturing image.")
#             break
#
#     # When everything is done, release the capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()


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


def single_draw_box(results):
    """
    :param yolo resutls object
    :param detect_class - the class number of the pick location
    :return: annotated image
    """
    global box

    box_colors = {"Green": (0, 255, 0), "Blue": (255, 0, 0), "Red": (0, 0, 255)}
    boxes_pick_class = results[0].boxes
    frame = results[0].orig_img

    for obj in enumerate(results[0].boxes):
        cls = obj[1].cls
        x1, y1, x2, y2 = obj[1].xyxy[0].tolist()
        if cls == 1:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_colors['Green'], thickness=3)
            text = "SCREW_PRESENT"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            # text_color = (255, 255, 255)  # White text
            # bg_color = (0, 0, 255)  # Red background

            # Get the size of the text box
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(frame, (int(x1), int(y1)-30), (int(x1)+text_width, int(y1)+text_height-10), box_colors['Green'], -1)
            cv2.putText(frame, f'SCEW_COMPLETE', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color=(0, 0, 0), thickness=thickness)

        else:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_colors['Red'], thickness=3)
            text = "SCREW_PRESENT"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            # text_color = (255, 255, 255)  # White text
            # bg_color = (0, 0, 255)  # Red background

            # Get the size of the text box
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(frame, (int(x1), int(y1) - 30), (int(x1) + text_width, int(y1) + text_height - 10),
                          box_colors['Red'], -1)
            cv2.putText(frame, f'SCEW_MISSING', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        color=(255, 255, 255), thickness=thickness)

    # for box in boxes_pick_class:
    #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_colors['Green'], thickness=3)
        # cv2.putText(frame, f'PICK THIS', (int(x1), int(y1) - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 255, 0), thickness=2)

    return frame


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
        x_center, y_center, box_width, box_height = box.xywh[0]

        # Convert YOLO format to pixel coordinates
        x_center = int(x_center)
        y_center = int(y_center)
        # box_width = int(width_rel * img_width)
        # box_height = int(height_rel * img_height)

        # Calculate bounding box corners
        x_min = int(x_center - box_width // 2)
        x_max = int(x_center + box_width // 2)
        y_min = int(y_center - box_height // 2)
        y_max = int(y_center + box_height // 2)

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

    return img


def main(camnum, weights_path: str, write_flag: bool):
    global results
    calibrate_camera = False
    undistort_image = False
    if calibrate_camera:
        ret, mtx, dist, rvecs, tvecs = calibrate_images(camnum)
        calibrate_camera = False  # Dont Change this value
        undistort_image = False  # Dont Change this value

    model = YOLO(weights_path)
    # frame_width = 480
    # frame_height = 480
    flag1 = False
    write_video = write_flag
    exp_name = 'ALEXA_DEMO'
    current_date = datetime.datetime.now()

    cap = cv2.VideoCapture(camnum)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # Write the frame to the video file
        # if flag1 and write_video:
        #     frame_width = int(frame.shape[1])
        #     frame_height = int(frame.shape[0])
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))
        #     directory = f'/Volumes/PycharmProjects/Assembly_Assist_Project/data/1.RawData/{exp_name}'
        #     if not os.path.exists(directory):
        #         # Create the directory
        #         os.makedirs(directory)
        #         print(f'Directory "{directory}" created.')
        #     else:
        #         print(f'Directory "{directory}" already exists.')
        #     file = f'{exp_name}_{current_date}.mp4'
        #     fullpath = os.path.join(directory, file)
        #     outmp4 = cv2.VideoWriter(fullpath, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))
        #     flag1 = False
        #
        # if success:
        #     if undistort_image:
        #         h, w = frame.shape[:2]
        #         newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        #         dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        #         x, y, w, h = roi
        #         frame = dst[y:y + h, x:x + w]
        results = model(frame, conf=0.6, verbose=False)
        annotated_frame = yolo_to_crosshair(frame,results[0].boxes)
        # annotated_frame = results[0].plot()
        cv2.imshow('Webcam Object Detection', annotated_frame)
        # if write_video:
        #     outmp4.write(annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cameraNumber = '/home/justin/Videos/Screencasts/Screencast from 2024-10-01 14-04-50.webm'
    weights_path = '/home/justin/PycharmProjects/DroneDetection/data/drone_test_results_09_30_242/weights/best.pt'
    write_video_flag = False
    main(cameraNumber, weights_path, write_video_flag)