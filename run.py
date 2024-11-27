import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

import chessboard
import detect


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)
state_list = []
pgn_list = []

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

# Function to check if hands are detected in a frame
def hands_detected(frame):
    # Convert the frame to RGB as MediaPipe uses RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get the result
    results = hands.process(rgb_frame)
    
    # If hands are detected, results.multi_hand_landmarks will not be None
    return results.multi_hand_landmarks is not None


def main(video_path):

    for index,frame in enumerate(frame_generator(video_path)):

        # if hand is there
        if not hands_detected(frame):

            detection = detect.detection(frame)
            detection_cg = {(piece_class,x+w//2,y+h) for piece_class,x,y,h,w in detection}

            # 
            img = chessboard.rotate_and_warp(frame)
            # also rotate the cg

            shape = img.shape
            
        


if __name__ == "__main__":
    # Replace with your video file path
    video_path = "2_move_student.mp4"
    main(video_path)
