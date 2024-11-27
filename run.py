import cv2
import numpy as np
import matplotlib.pyplot as plt



def frame_generator(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        yield frame  # Yield frame to the caller

    cap.release()

def main(video_path):


if __name__ == "__main__":
    # Replace with your video file path
    video_path = "2_move_student.mp4"
    main(video_path)
