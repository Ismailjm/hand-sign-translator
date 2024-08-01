import os
import cv2

def flip_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            # Flip the image horizontally
            flipped_img = cv2.flip(img, 1)

            # Save the flipped image with a new name
            flipped_img_path = os.path.join(folder_path, "flipped_" + filename)
            cv2.imwrite(flipped_img_path, flipped_img)

def process_all_classes(data_folder):
    for class_folder in os.listdir(data_folder):
        class_folder_path = os.path.join(data_folder, class_folder)
        if os.path.isdir(class_folder_path):
            flip_images_in_folder(class_folder_path)

# Provide the path to your 'data' folder here
data_folder = "./data"

process_all_classes(data_folder)
