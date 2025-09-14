import time
import cv2

# index = 0
# while True:
#     cap = cv2.VideoCapture(index)
#     if not cap.isOpened():
#         print(f"Camera index {index} not available.")
#         break

#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow(f"Camera {index}", frame)
#         cv2.waitKey(1000)  # Display each camera feed for 1 second
#         cv2.destroyAllWindows()
#         height, width, _ = frame.shape
#         print(f"Found camera at index {index}: {width}x{height}")
#     else:
#         print(f"Could not read frame from camera at index {index}, but it exists.")

#     time.sleep(10)
#     cap.release()
#     index += 1

# Replace with the index you found for your iPhone
IPHONE_CAMERA_INDEX = 0

cap = cv2.VideoCapture(IPHONE_CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {IPHONE_CAMERA_INDEX}")
    exit()

# Optional: Set a lower resolution to speed up processing
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Successfully opened iPhone camera. Press 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
        
    # Your inference/analysis code would go here
    # For now, we'll just display the live feed
    
    cv2.imshow('iPhone Live Feed', frame)
    
    # Wait for the 'q' key to be pressed to exit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print("Stream closed.")