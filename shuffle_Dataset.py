"""
Justin Clay
May 29 2024
Dataset Shuffle Tool for Yolo V1.1 Outputs on CVAT. This tool will randomly shuffle and build the file structure for Yolo V5 input
"""
import os
import re
import numpy as np
import shutil

# folder = '/home/justin/PycharmProjects/Assembly_Assist_Project/data/4.CombinedDatasets/ALEXA_DEMO_V2_ABC'
train_split = 0.8


def list_files(directory):
    """
    List all file paths in the given directory and its subdirectories.

    Parameters:
    directory (str): The directory to list files from.

    Returns:
    list: A list of file paths.
    """
    file_paths = []

    # Walk through directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Join the root directory and file name to get the full path
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths


def remove_files(file_list):
    list_data = []
    for i in file_list:
        if i.split("/")[-1] == '.DS_Store' or i.split("/")[-1] == 'train.txt':
            # print('skip')
            continue
        if i.split('/')[-2] == 'obj_train_data':
            list_data.append(i)
            continue
        list_data.append(i)
    final_list = np.array(list_data)
    return final_list


def get_extension(filename):
    return filename.split('.')[-1]


def change_extension_and_check(filepath):
    extensions = ['.PNG','.png','.jpg','.JPG','.JPEG']
    for ext in extensions:
        img_file = filepath.replace('.txt', ext)
        if os.path.isfile(img_file):
            return img_file
        else:
            continue
    print(f'File Extension Not Found:{extensions}')


def generate_structure(file_list, folder):
    global train_split, files, train_list, val_list
    yolo_folder_structure = ['images/train', 'images/val', 'labels/train', 'labels/val']
    [os.makedirs(os.path.join(folder, directory), exist_ok=True) for directory in yolo_folder_structure]
    files = np.array(file_list).squeeze()
    files_filtered = np.array([filename for filename in files if filename.endswith('.txt')])
    files_shuffled = np.random.permutation(files_filtered)
    n_files = files_shuffled.shape[0]
    train_list = files_shuffled[0:int(train_split * n_files)]
    val_list = files_shuffled[int(train_split * n_files):]
    for traindata in train_list:
        move_file(traindata, os.path.join(folder, yolo_folder_structure[2]))
        img_file = change_extension_and_check(traindata)
        # img_file = i.replace('.txt', '.PNG')
        move_file(img_file, os.path.join(folder, yolo_folder_structure[0]))
    for valdata in val_list:
        move_file(valdata, os.path.join(folder, yolo_folder_structure[3]))
        # img_file = valdata.replace('.txt', '.PNG')
        img_file = change_extension_and_check(valdata)
        move_file(img_file, os.path.join(folder, yolo_folder_structure[1]))


def move_file(source, destination):
    """
        # Example usage
        source_path = 'path/to/your/source/file.txt'
        destination_path = 'path/to/your/destination/directory/file.txt'
        move_file(source_path, destination_path)
    """
    try:
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Move the file
        shutil.move(source, destination)
        print(f"File moved from {source} to {destination}")
    except FileNotFoundError:
        print(f"The source file '{source}' does not exist.")
        pass
    except PermissionError:
        print(f"Permission denied to move the file '{source}' to '{destination}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        pass


import os


def check_directory_structure(root_dir, expected_structure):
    """
    Check if the directory has the expected structure.

    :param root_dir: The root directory to check.
    :param expected_structure: A dictionary representing the expected directory structure.
                               Keys are directory names, and values can either be lists of expected subdirectories or files.
    :return: True if the directory structure matches, False otherwise.
    """
    for dir_name, sub_items in expected_structure.items():
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path):
            return False

        if isinstance(sub_items, dict):
            # Recursively check for subdirectories and their expected content
            if not check_directory_structure(dir_path, sub_items):
                return False
        else:
            for item in sub_items:
                item_path = os.path.join(dir_path, item)
                if not os.path.exists(item_path):
                    return False

    return True


shuffle_list = [
'/home/justin/PycharmProjects/DroneDetection/data/drone_dataset2 (Copy)'
]


for i in shuffle_list:

# Yolo folder Structure
    expected_structure = {
        "images": {
            "train": {},
            "val": {}
        },
        "labels": {
            "train": {},
            "val": {}
        }
    }
    if check_directory_structure(i, expected_structure):
        print(f"WARNING:Directory structure is already shuffled, skipping :{i}")
        continue
    else:
        print("Starting Shuffle")
        listoffiles = list_files(i)
        list_paths = remove_files(listoffiles)
        generate_structure(list_paths, i)