import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

mp_hands = mp.solutions.hands

DATA_DIR = './data'

def process_image(args):
    dir_, img_path = args
    # Each thread needs its own Hands instance (not thread-safe to share)
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


# Build task list
tasks = [
    (dir_, img_path)
    for dir_ in os.listdir(DATA_DIR)
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))
]

data = []
labels = []

with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(process_image, task): task for task in tasks}
    for future in as_completed(futures):
        result, label = future.result()
        if result is not None:
            data.append(result)
            labels.append(label)

print(labels)
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)