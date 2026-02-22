import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

devnull_fd = os.open(os.devnull, os.O_WRONLY)
old_stderr_fd = os.dup(2)
os.dup2(devnull_fd, 2)
os.close(devnull_fd)

import sys
import pickle
import mediapipe as mp
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

mp_hands = mp.solutions.hands
DATA_DIR = './data'

def process_image(args):
    dir_, img_path = args
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
    if img is None:
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hands.close()

    if not results.multi_hand_landmarks:
        return None, None

    data_aux = []
    for hand_landmarks in results.multi_hand_landmarks:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        mins = coords.min(axis=0)
        normalized = coords - mins
        data_aux.extend(normalized.flatten().tolist())

    return data_aux, dir_


tasks = [
    (dir_, img_path)
    for dir_ in os.listdir(DATA_DIR)
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))
]

data = []
labels = []

with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(process_image, task): task for task in tasks}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images", file=sys.stdout):
        result, label = future.result()
        if result is not None:
            data.append(result)
            labels.append(label)

os.dup2(old_stderr_fd, 2)
os.close(old_stderr_fd)

print(labels)
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)