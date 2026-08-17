import os
import csv
import copy
import argparse
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class MockHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = [MockLandmark(lm.x, lm.y, lm.z) for lm in landmarks]

class MockClassification:
    def __init__(self, label):
        self.label = label

class MockHandedness:
    def __init__(self, categories):
        self.classification = [MockClassification(categories[0].category_name)]

class ProcessResult:
    def __init__(self, multi_hand_landmarks, multi_handedness):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness

class HandsWrapper:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ):
        base_options = python.BaseOptions(model_asset_path='model/keypoint_classifier/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        res = self.detector.detect(mp_image)
        
        multi_hand_landmarks = []
        multi_handedness = []
        
        if res.hand_landmarks:
            for lms in res.hand_landmarks:
                multi_hand_landmarks.append(MockHandLandmarks(lms))
            for hand in res.handedness:
                multi_handedness.append(MockHandedness(hand))
        
        if not multi_hand_landmarks:
            return ProcessResult(None, None)
            
        return ProcessResult(multi_hand_landmarks, multi_handedness)

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


datasetdir = "model/dataset/dataset 1"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    return args


class PythonPredictionStabilizer:
    def __init__(self, alpha=0.45, switch_thresh=0.42, hold_thresh=0.22, debounce_frames=2):
        self.alpha = alpha
        self.switch_thresh = switch_thresh
        self.hold_thresh = hold_thresh
        self.debounce_frames = debounce_frames
        self.smoothed_probs = np.zeros(26, dtype=np.float32)
        self.current_idx = None
        self.candidate_idx = None
        self.candidate_count = 0
        self.is_init = False

    def update(self, raw_probs):
        if not self.is_init:
            self.smoothed_probs = np.array(raw_probs, dtype=np.float32)
            self.is_init = True
        else:
            self.smoothed_probs = self.alpha * np.array(raw_probs, dtype=np.float32) + (1.0 - self.alpha) * self.smoothed_probs

        top_idx = int(np.argmax(self.smoothed_probs))
        top_prob = float(self.smoothed_probs[top_idx])

        if self.current_idx is None:
            if top_prob >= self.switch_thresh:
                self.current_idx = top_idx
                self.candidate_idx = None
                self.candidate_count = 0
        elif self.current_idx == top_idx:
            self.candidate_idx = None
            self.candidate_count = 0
        else:
            current_prob = float(self.smoothed_probs[self.current_idx])
            if self.candidate_idx == top_idx:
                self.candidate_count += 1
            else:
                self.candidate_idx = top_idx
                self.candidate_count = 1

            should_switch = (top_prob >= self.switch_thresh and self.candidate_count >= self.debounce_frames) or \
                            (current_prob < self.hold_thresh and top_prob > current_prob + 0.15)
            if should_switch:
                self.current_idx = top_idx
                self.candidate_idx = None
                self.candidate_count = 0

        conf = float(self.smoothed_probs[self.current_idx]) if self.current_idx is not None else 0.0
        return self.current_idx, conf

    def reset(self):
        self.current_idx = None
        self.candidate_idx = None
        self.candidate_count = 0
        self.is_init = False
        self.smoothed_probs.fill(0)


class PythonHeadGestureDetector:
    def __init__(self, cooldown_ms=650, time_window_ms=850):
        self.cooldown_ms = cooldown_ms
        self.time_window_ms = time_window_ms
        self.last_trigger_time = 0
        self.history = []
        self.current_state = 'CENTER'

    def update_from_landmarks(self, landmark_list, timestamp_ms):
        if landmark_list is None or len(landmark_list) < 21:
            return False
            
        if timestamp_ms - self.last_trigger_time < self.cooldown_ms:
            return False

        wrist = landmark_list[0]
        middle_mcp = landmark_list[9]
        index_mcp = landmark_list[5]
        pinky_mcp = landmark_list[17]

        palm_width = np.hypot(pinky_mcp[0] - index_mcp[0], pinky_mcp[1] - index_mcp[1])
        if palm_width < 5:
            return False

        dx = middle_mcp[0] - wrist[0]
        tilt_ratio = dx / float(palm_width)

        new_state = 'CENTER'
        if tilt_ratio < -0.45:
            new_state = 'LEFT'
        elif tilt_ratio > 0.45:
            new_state = 'RIGHT'

        if new_state != self.current_state:
            if new_state in ('LEFT', 'RIGHT'):
                self.history.append((new_state, timestamp_ms))
                self.history = [h for h in self.history if timestamp_ms - h[1] <= self.time_window_ms]
            self.current_state = new_state

        if len(self.history) >= 2:
            states = [h[0] for h in self.history]
            if 'LEFT' in states and 'RIGHT' in states:
                self.last_trigger_time = timestamp_ms
                self.history = []
                self.current_state = 'CENTER'
                return True

        return False


def smooth_landmarks(prev_landmarks, current_landmarks, alpha=0.72):
    if prev_landmarks is None or len(prev_landmarks) != len(current_landmarks):
        return current_landmarks
    smoothed = []
    for (px, py), (cx, cy) in zip(prev_landmarks, current_landmarks):
        sx = int(alpha * cx + (1 - alpha) * px)
        sy = int(alpha * cy + (1 - alpha) * py)
        smoothed.append([sx, sy])
    return smoothed


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    hands = HandsWrapper(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    stabilizer = PythonPredictionStabilizer()
    prev_landmarks_dict = {}

    # Read labels ###########################################################
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if mode == 2:
            # Check if dataset directory exists
            if not os.path.exists(datasetdir):
                print(f"Dataset directory '{datasetdir}' not found.")
                mode = 0
                continue

            # Loading image while processing the dataset
            loading_img = cv.imread("./assets/om606.png", cv.IMREAD_COLOR)
            if loading_img is not None:
                cv.putText(
                    loading_img,
                    "Loading...",
                    (20, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    4,
                    cv.LINE_AA,
                )
                cv.imshow("Hand Gesture Recognition", loading_img)
                key = cv.waitKey(1000)

            # Looping through each folder of the dataset
            imglabel = -1
            for imgclass in os.listdir(datasetdir):
                imglabel += 1
                numofimgs = 0
                class_path = os.path.join(datasetdir, imgclass)
                if not os.path.isdir(class_path):
                    continue
                for img_name in os.listdir(class_path):
                    numofimgs += 1
                    imgpath = os.path.join(class_path, img_name)
                    try:
                        img = cv.imread(imgpath)
                        if img is None:
                            continue
                        debug_img = copy.deepcopy(img)

                        for _ in [1, 2]:
                            img.flags.writeable = False
                            results = hands.process(img)
                            img.flags.writeable = True

                            if results.multi_hand_landmarks is not None:
                                for hand_landmarks, handedness in zip(
                                    results.multi_hand_landmarks,
                                    results.multi_handedness,
                                ):
                                    # Bounding box calculation
                                    brect = calc_bounding_rect(
                                        debug_img, hand_landmarks
                                    )
                                    # Landmark calculation
                                    landmark_list = calc_landmark_list(
                                        debug_img, hand_landmarks
                                    )

                                    # Conversion to relative coordinates / normalized coordinates
                                    pre_processed_landmark_list = pre_process_landmark(
                                        landmark_list
                                    )

                                    # Write to the dataset file
                                    logging_csv(
                                        imglabel, mode, pre_processed_landmark_list
                                    )
                            img = cv.flip(img, 0)
                    except Exception as e:
                        print(f"Issue with image {imgpath}: {e}")

                print(f"Num of image of the class {imglabel} is : {numofimgs}")
            mode = 1
            print("End of job!")
            break
        else:
            if results.multi_hand_landmarks is not None:
                for h_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # Landmark calculation
                    raw_landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    
                    # Smooth landmark points
                    smoothed_landmark_list = smooth_landmarks(
                        prev_landmarks_dict.get(h_idx), raw_landmark_list
                    )
                    prev_landmarks_dict[h_idx] = smoothed_landmark_list

                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(raw_landmark_list)

                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list)

                    # Hand sign classification with probabilities & stabilizer
                    raw_probs = keypoint_classifier.predict_probabilities(pre_processed_landmark_list)
                    hand_sign_id, conf = stabilizer.update(raw_probs)

                    label_text = ""
                    if hand_sign_id is not None and conf >= 0.35 and hand_sign_id < len(keypoint_classifier_labels):
                        label_text = f"{keypoint_classifier_labels[hand_sign_id]} ({int(conf * 100)}%)"

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, smoothed_landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        label_text,
                    )
            else:
                stabilizer.reset()
                prev_landmarks_dict.clear()

            debug_image = draw_info(debug_image, fps, mode, number)

            # Screen reflection #############################################################
            cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 65 <= key <= 90:  # A ~ B
        number = key - 65
    if key == 110:  # n (Inference Mode)
        mode = 0
    if key == 107:  # k (Capturing Landmark From Camera Mode)
        mode = 1
    if key == 100:  # d (Capturing Landmarks From Provided Dataset Mode)
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

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


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if (mode == 1 or mode == 2) and (0 <= number <= 35):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (255, 255, 255),
            2,
        )

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[6]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_point[7]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_point[8]),
            (255, 255, 255),
            2,
        )

        # Middle finger
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[10]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[10]),
            tuple(landmark_point[11]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[11]),
            tuple(landmark_point[12]),
            (255, 255, 255),
            2,
        )

        # Ring finger
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 255, 255),
            2,
        )

        # Little finger
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (255, 255, 255),
            2,
        )

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (255, 255, 255),
            2,
        )

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    return image


def draw_info(image, fps, mode, number):
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    mode_string = [
        "Logging Key Point",
        "Capturing Landmarks From Provided Dataset Mode",
    ]
    if 1 <= mode <= 2:
        cv.putText(
            image,
            "MODE:" + mode_string[mode - 1],
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                "NUM:" + str(number),
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    return image


if __name__ == "__main__":
    main()
