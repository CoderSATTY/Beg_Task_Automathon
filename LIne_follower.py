import cv2
import numpy as np


video_path = '/home/satyam-ashtikar/Python/LINE_FOLLOWER_PS(BEGINNER).mp4'
cap = cv2.VideoCapture(video_path) 

if not cap.isOpened():
        print("Error: Could not open camera.")
        

while True:
        # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    height, width = frame.shape[:2]
    roi = frame[height-100:height, :]  # Last 100 rows (bottom part)


        # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to isolate the black line
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the black line
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
            # Find the largest contour (assumed to be the black line)
        largest_contour = max(contours, key=cv2.contourArea)

            # Get the center of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
                # Draw the contour and center point
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # Print the center coordinates (for control purposes)
            print(f"Center of line: ({cx}, {cy})")

        # Display the frames
    rsz_frm=cv2.resize(roi, (0,0),fx=0.5,fy=0.5)
    rsz_bin=cv2.resize(binary, (0,0),fx=0.5,fy=0.5)
    cv2.imshow('Frame', roi)
    cv2.imshow('Binary', rsz_bin)

        # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
