import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from collections import Counter
import chess

import chessboard
import detect

# initaiate hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)
# a to h from left to right
X_INDEX = list("abcdefgh")
# 8 to 1 from top to bottom
Y_INDEX = [str(i+1) for i in range(8)][::-1]
# initiate state accumulation variables
state_list = []
pgn_list = []
# noise frame tolerance
TOLERANCE = 10
# black class name
BLACK = "black"
WHITE = "white"

# wrap the video reader
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

# summarize state list into easy to use format
def summarize_states(lst, tolerance,verbose=False):
    if verbose:
        print("Original list of states:", lst)
    
    # Step 1: Count occurrences of each exact state
    state_counts = Counter(frozenset(state) for state in lst)
    if verbose:
        print("State counts:", state_counts)
    
    # Determine major states (those appearing >= tolerance times)
    major_states = {state for state, count in state_counts.items() if count >= tolerance}
    if verbose:
        print("Major states:", major_states)
    
    # Filter list to include only major states
    filtered_lst = [state for state in lst if frozenset(state) in major_states]
    if verbose:
        print("Filtered list:", filtered_lst)

    # Step 2: Summarize consecutive phases
    summarized_states = []
    for state in filtered_lst:
        if not summarized_states or summarized_states[-1] != state:
            summarized_states.append(state)
    if verbose:
        print("Summarized states:", summarized_states)

    # Step 3: Identify differences between consecutive states
    differences = []
    for i in range(len(summarized_states) - 1):
        old_state = summarized_states[i]
        new_state = summarized_states[i + 1]
        
        disappeared = old_state - new_state
        appeared = new_state - old_state
        
        differences.append((disappeared, appeared))
    if verbose:
        print("Differences:", differences)

    return differences

# TODO: Add any weird behavior apart from capturing
def pgn_from_differences(differences):

    KEY = {"knight":"N","king":"K","bishop":"B","queen":"Q","rook":"R","pawn":""}
    # KEY_CAPTURE = {"knight":"N","king":"K","bishop":"B","queen":"Q","rook":"R","pawn":"P"}

    """
    Convert the output of summarize_states into PGN format.

    Parameters:
        differences (list): List of differences as ({disappeared}, {appeared}).

    Returns:
        list: A list of PGN strings describing the moves.
    """
    pgn_moves = []

    for disappeared, appeared in differences:

        temp = []

        # Match disappeared and appeared pieces
        for old_piece in disappeared:
            # Try to find a matching appeared piece with the same (color, class)
            match = next((new_piece for new_piece in appeared 
                          if old_piece[:2] == new_piece[:2]), None)
            if match:
                # Movement: piece moved to a new position
                old_pos = f"{old_piece[2]}{old_piece[3]}"
                new_pos = f"{match[2]}{match[3]}"
                # Add movement using the correct piece name
                temp.append((KEY[old_piece[1]],old_pos,new_pos))
            else:
                # Piece disappeared without a matching appearance (elimination or capture)
                old_pos = f"{old_piece[2]}{old_piece[3]}"
                temp.append(f"{KEY[old_piece[1]]}{old_pos}x")
        if len(temp) == 2:
            if temp[0][0] != "":
                pgn_moves.append(f"{temp[0][0]}x{temp[0][2]}")
            else:
                pgn_moves.append(f"{temp[0][1][0]}x{temp[0][2]}")
        else:
            pgn_moves.append(f"{temp[0][0]}{temp[0][2]}")

    # Group into pairs of black and white
    def group_into_pairs(lst):
        # Group consecutive elements in pairs, leave the last element if it's odd
        return [lst[i] + " " + lst[i + 1] if i + 1 < len(lst) else lst[i] for i in range(0, len(lst), 2)]

    pgn = [f"{i+1}. {elm}" for i, elm in enumerate(group_into_pairs(pgn_moves))]

    return pgn

def visualization(ori, pgn):

    board = chess.Board()
    board.clear()
    for color, piece, alpha, num in ori:
        
        if color == BLACK:
            piece = piece.lower()
        elif color == WHITE:
            piece = piece.upper()

        position = alpha + num
        square = chess.parse_square(position)
        board.set_piece_at(square, chess.Piece.from_symbol(piece))

    state_list = list()
    state_list.append(str(board))

    for line in pgn:
        num, move1, move2 = pgn.split()

        board.push_san(move1)
        state_list.append(str(board))

        board.push_san(move2)
        state_list.append(str(board))

    return state_list

# main
def main(video_path):

    for index,frame in enumerate(frame_generator(video_path)):

        # if hand is there
        if not hands_detected(frame):

            # detect pieces and format the piece's foot CG
            detection = detect.detection(frame)
            detection_cg = {(piece_class,x+w//2,y+h) for piece_class,x,y,h,w in detection}

            # get the transformed image and the coordinate of the transformed CG
            img, piece_cg = chessboard.rotate_and_warp(frame,detection_cg)

            # get the image size to divide into cells
            shape = img.shape
            x_cell_size = shape[0]//8
            y_cell_size = shape[1]//8

            # reformat the piece_cg set to indicate row and column instead
            detection_cell = {(piece_class.split("-")[0], # color
                               piece_class.split("-")[0], # class
                               X_INDEX[x//x_cell_size], # column: a,b,c..
                               Y_INDEX[y//y_cell_size]) # row: 1,2,...
                               for piece_class,x,y in piece_cg}
            
            state_list.append(detection_cell)

    # Original state
    ori = state_list[0]
        
    # Once the processing is finished
    differences = summarize_states(lst=state_list, tolerance=TOLERANCE)
    pgn = pgn_from_differences(differences)
    board_states = visualization(ori, pgn)

    return pgn, board_states   


if __name__ == "__main__":
    # Replace with your video file path
    video_path = "2_move_student.mp4"
    pgn, board_states = main(video_path)
    print(pgn)
