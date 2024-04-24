import cv2
import imutils

# Define the range of red color in HSV
redLower = (0, 120, 70)
redUpper = (10, 255, 255)

# Initialize the webcam
camera = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    (grabbed, frame) = camera.read()
    
    # Resize the frame to improve processing speed
    frame = imutils.resize(frame, width=1000)
    
    # Blur the frame to reduce noise and convert it to HSV color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate red regions in the frame
    mask = cv2.inRange(hsv, redLower, redUpper)
    
    # Perform erosion and dilation to remove noise from the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Initialize variables for object position and radius
    center = None
    
    # If contours are found
    if len(cnts) > 0:
        # Find the largest contour and its enclosing circle
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Calculate the moments of the contour to find the centroid
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Draw the circle and centroid on the frame
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Determine the direction based on object position
            if radius > 250:
                print("Stop")
            else:
                if center[0] < 150:
                    print("Right")
                elif center[0] > 450:
                    print("Left")
                elif radius < 250:
                    print("Front")
                else:
                    print("Stop")
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Check for key press to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
