import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load your trained model
model_dict = pickle.load(open('./app/Python/model_svm.p', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']  # Don't forget to load the scaler!

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []
        
        # IMPORTANT: Only process the first 2 hands detected
        hands_to_process = results.multi_hand_landmarks[:2]
        
        # Collect landmarks from detected hands (max 2)
        for hand_landmarks in hands_to_process:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Collect all landmarks for this hand
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
        
        # Normalize coordinates relative to minimum values
        for i in range(len(x_)):
            data_aux.append(x_[i] - min(x_))
            data_aux.append(y_[i] - min(y_))
        
        # Pad with zeros if only one hand is detected
        if len(data_aux) < 84:
            data_aux.extend([0.0] * (84 - len(data_aux)))
        
        # Ensure we don't exceed 84 features
        data_aux = data_aux[:84]
        
        # IMPORTANT: Scale the data before prediction
        data_scaled = scaler.transform([np.asarray(data_aux)])
        
        # Make prediction (without probability)
        prediction = model.predict(data_scaled)
        predicted_character = prediction[0]
        
        # Display prediction on frame
        num_hands = len(hands_to_process)
        
        cv2.putText(frame, f"Prediction: {predicted_character}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Hands detected: {num_hands}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No hands detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Sign Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()