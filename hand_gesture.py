# Importing necessary libraries
import cv2
import numpy as np

# Defining a function to detect and classify hand gestures
def hand_gesture_recognition():
    # Open a video capture object (use 0 for default webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to exit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Flip the frame horizontally to get mirror image
        frame = cv2.flip(frame, 1)
        
        # Define region of interest (ROI) where the hand gesture will be detected
        roi = frame[100:400, 300:600]
        
        # Draw a rectangle around the ROI
        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 2)
        
        # Convert ROI to HSV color space
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define the range for skin color detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Threshold the HSV image to get only skin colors
        mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        
        # Apply GaussianBlur to smoothen the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour based on area
            max_contour = max(contours, key=cv2.contourArea)
            
            # Draw the contour on the ROI
            cv2.drawContours(roi, [max_contour], -1, (0, 0, 255), 2)
            
            # Calculate the convex hull of the contour
            hull = cv2.convexHull(max_contour)
            
            # Find the defects in the convex hull
            hull_indices = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull_indices)
            
            # Count the number of fingers based on defects
            finger_count = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])
                    
                    # Calculate distance between start and far point
                    a = np.linalg.norm(np.array(start) - np.array(far))
                    # Calculate distance between end and far point
                    b = np.linalg.norm(np.array(end) - np.array(far))
                    # Calculate distance between start and end point
                    c = np.linalg.norm(np.array(start) - np.array(end))
                    
                    # Calculate angle using cosine rule
                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                    
                    # Count fingers if the angle is below 90 degrees
                    if angle <= np.pi / 2:
                        finger_count += 1
            
            # Display the number of fingers detected
            cv2.putText(frame, f"Fingers: {finger_count + 1}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the processed frames
        cv2.imshow("Hand Gesture Recognition", frame)
        cv2.imshow("Skin Mask", mask)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

# Run the program
hand_gesture_recognition()
