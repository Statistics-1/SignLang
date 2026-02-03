import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'app\Python\Data&modeltraining\data\ASL_Dataset\Train'

data = []
labels = []

# Count total images first for accurate progress bar
total_images = 0
for dir_ in os.listdir(DATA_DIR):
    total_images += len(os.listdir(os.path.join(DATA_DIR, dir_)))

# Process images with progress bar
with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)
            
            pbar.update(1)

print(f"\nProcessed {len(data)} images with hand landmarks detected")
print(f"Saving to data.pickle...")

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("Done!")