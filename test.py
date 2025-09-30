import cv2

index = 0
while True:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        break
    print(f"Camera index {index}: Found")
    cap.release()
    index += 1

if index == 0:
    print("No cameras found.")
