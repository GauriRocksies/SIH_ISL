from function import *
import cv2
import numpy as np
import os
from time import sleep

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')


actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
no_sequences = 80
sequence_length = 80

# Create directories if they don't exist
for action in actions: 
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                # Construct path to frame
                frame_path = f'C://Users//Asus//OneDrive//Desktop//sih//Image//{action}//{sequence}.png'
                
                # Print the path for debugging
                print(f"Trying to load frame from: {frame_path}")
                
                # Read feed
                frame = cv2.imread(frame_path)
                
                # Check if frame is loaded
                if frame is None:
                    print(f"Failed to load frame: {frame_path}")
                    continue  # Skip this frame if it cannot be loaded
                
                # Make detections
                image, results = mediapipe_detection(frame, hands)
                
                if results is None:
                    print("Skipping this frame due to detection issue.")
                    continue  # Skip this frame if processing failed

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else: 
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Show the frame with annotations
                cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + '.npy')
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cv2.destroyAllWindows()