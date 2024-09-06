import os
import cv2

url="http://192.168.0.108:4747/video"
cap = cv2.VideoCapture(url)
directory = 'C://Users//Asus//OneDrive//Desktop//sih//Image'


# Ensure subdirectories exist
subfolders = [chr(i) for i in range(ord('A'), ord('Z')+1)]
for folder in subfolders:
    os.makedirs(os.path.join(directory, folder), exist_ok=True)

while True:
    _, frame = cap.read()

    # Count the number of images in each subdirectory
    count = {letter.lower(): len(os.listdir(os.path.join(directory, letter))) for letter in subfolders}

    # Display ROI and frame
    row = frame.shape[1]
    col = frame.shape[0]
    
    cv2.rectangle(frame,(0,40),(300,400),(255,255,255),2)
    cv2.imshow("data",frame)
    cv2.imshow("ROI",frame[40:400,0:300])
    frame=frame[40:400,0:300]
    # interrupt = cv2.waitKey(10)

    # Get the key pressed
    interrupt = cv2.waitKey(10) 

    # Check for key press and save the corresponding frame
    if interrupt != -1 and 0 <= interrupt <= 0x10FFFF:
        if 'a' <= chr(interrupt).lower() <= 'z':
            letter = chr(interrupt).upper()
            filepath = os.path.join(directory, letter, f'{count[letter.lower()]}.png')
            cv2.imwrite(filepath, frame[40:400, 0:300])

    # Break the loop when '1' is pressed
    if interrupt == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()