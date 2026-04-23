import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load model
model_dict = pickle.load(open('modelrf.p', 'rb'))
model = model_dict['model']

label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

CONFIDENCE_THRESHOLD = 0.1
MAX_FEATURES = 126  # 2 hands × 21 landmarks × 3 coords (x, y, z)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []

        hands_to_process = results.multi_hand_landmarks[:2]

        for hand_landmarks in hands_to_process:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Normalize each hand independently (matches training)
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            coords -= coords.min(axis=0)
            data_aux.extend(coords.flatten().tolist())

        # Pad to 126 if only 1 hand detected
        if len(data_aux) < MAX_FEATURES:
            data_aux.extend([0.0] * (MAX_FEATURES - len(data_aux)))

        data_aux = data_aux[:MAX_FEATURES]

        prediction = model.predict([np.asarray(data_aux)])
        prediction_proba = model.predict_proba([np.asarray(data_aux)])

        predicted_character = int(prediction[0])
        confidence = np.max(prediction_proba)
        num_hands = len(hands_to_process)

        if confidence >= CONFIDENCE_THRESHOLD:
            cv2.putText(frame, f"Prediction: {label_map[predicted_character]}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Unknown sign", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3, cv2.LINE_AA)

        cv2.putText(frame, f"Hands detected: {num_hands}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No hands detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hand Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
