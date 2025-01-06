import cv2

# Step 1: Added the video path and captured it using VideoCapture function...
video_path = '/home/satyam-ashtikar/Python/LINE_FOLLOWER_PS(BEGINNER).mp4'
cap = cv2.VideoCapture(video_path) 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.45)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.45)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_path = '/home/satyam-ashtikar/Python/processed_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))  # *2 for concatenated frame width

# Step 2: Took each Frame of the video and added functionality to that frame for processing it that's why we apply while loop to load frames untill frames are returned...
while True:
    ret, frame = cap.read() # ret is set to True when the frames are read

    if not ret: # If frame was not read, ret is set to False and we exit the loop
        print("Image was not returned")
        break
    
    # Converting the BGR frame to Grayscale frame...
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applied Binary Thresholding to convert the Grayscale image to binary image which has only Black and White values...
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

    # This will find contours but only the outermost ones...
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # The black line path will be the contour with the Maximum area as we are neglecting the edges...
        largest_contour = max(contours, key=cv2.contourArea)

        # Centroid calculation using moments
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:  # Avoiding division by zero
            cx = int(M["m10"] / M["m00"])  # X-coordinate of the centroid
            cy = int(M["m01"] / M["m00"])  # Y-coordinate of the centroid

        # Draw the centroid on the image...
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Resizing width and height by half...
    rsz_frm=cv2.resize(frame, (0,0),fx=0.45,fy=0.45)
    rsz_bin=cv2.resize(binary_img, (0,0),fx=0.45,fy=0.45)

    # Converting Grayscale frame to BGR frame to use cv2, hconcat as Grayscale images has only 1 colour channel while BGR images has 3 colour channels...
    rsz_bin_bgr = cv2.cvtColor(rsz_bin, cv2.COLOR_GRAY2BGR)

    # Concatenating frames horizontally...
    combined_frame = cv2.hconcat([rsz_frm, rsz_bin_bgr])

    # Write the frame to the output video
    out.write(combined_frame)

    # Displaying the combined frame :)
    cv2.imshow('Contoured and Thresholded Video', combined_frame)
 
    # Checking the key press for every 1ms and we can exit the loop if 'q' is pressed...
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

# Step 3: Releasing the video capture object and closing the OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()