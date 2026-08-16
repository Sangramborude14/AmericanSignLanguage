import os
import cv2 as cv
import numpy as np
import copy
import itertools
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATASET_DIR = r"C:\Users\Sangram Borude\.cache\kagglehub\datasets\ayuraj\asl-dataset\versions\1\asl_dataset"
OUTPUT_CSV = "model/keypoint_classifier/keypoint_new.csv"
MODEL_ASSET_PATH = "model/keypoint_classifier/hand_landmarker.task"

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0:
        max_value = 1.0

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    class_map = {letter: idx for idx, letter in enumerate(alphabet)}

    # Open CSV for writing
    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        total_processed = 0
        total_extracted = 0

        # Loop through a-z folders
        for letter in alphabet:
            letter_dir = os.path.join(DATASET_DIR, letter)
            if not os.path.isdir(letter_dir):
                print(f"Directory {letter_dir} not found. Skipping.")
                continue

            class_idx = class_map[letter]
            print(f"Processing folder '{letter}' (label: {class_idx})...")

            for file_name in os.listdir(letter_dir):
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(letter_dir, file_name)
                total_processed += 1

                # Load image
                image = cv.imread(img_path)
                if image is None:
                    continue

                # Process hand landmarks
                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                res = detector.detect(mp_image)

                if res.hand_landmarks:
                    landmarks = res.hand_landmarks[0]
                    landmark_list = calc_landmark_list(image, landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    # Write to CSV
                    writer.writerow([class_idx, *pre_processed_landmark_list])
                    total_extracted += 1

            print(f"Finished folder '{letter}'. Accumulated {total_extracted} records so far.")

    print(f"\nFeature extraction complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Total feature vectors extracted and saved to {OUTPUT_CSV}: {total_extracted}")

if __name__ == "__main__":
    main()
