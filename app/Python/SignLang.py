import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load your trained model
model_dict = pickle.load(open('./app/Python/model.p', 'rb'))
model = model_dict['model']

# Create label mapping dictionary (outside the loop)
newlable = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

cap = cv2.VideoCapture(0)

# Set confidence threshold (adjust this value between 0.0 and 1.0)
CONFIDENCE_THRESHOLD = 0.7

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
        # Each hand has 21 landmarks * 2 coordinates = 42 features
        # Total expected features = 84 (for 2 hands)
        if len(data_aux) < 84:
            # Pad with zeros to reach 84 features
            data_aux.extend([0.0] * (84 - len(data_aux)))
        
        # Ensure we don't exceed 84 features (safety check)
        data_aux = data_aux[:84]
        
        # Make prediction with probability
        prediction = model.predict([np.asarray(data_aux)])
        prediction_proba = model.predict_proba([np.asarray(data_aux)])
        
        predicted_character = int(prediction[0])
        confidence = np.max(prediction_proba)  # Get the highest probability
        
        # Display prediction on frame
        num_hands = len(hands_to_process)
        
        # Only show prediction if confidence is above threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            cv2.putText(frame, f"Prediction: {newlable[predicted_character]}", (50, 50), 
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