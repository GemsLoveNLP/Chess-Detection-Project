import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from collections import Counter
import chess
import chess.pgn

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
# class names
BLACK = "black"
WHITE = "white"
KING = "king"
# promotion key for uci
KEY = {"knight":"n","king":"k","bishop":"b","queen":"q","rook":"r","pawn":""}

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

def uci_from_differences(differences):

    """
    Convert the output of summarize_states into UCI format.

    Parameters:
        differences (list): List of differences as ({disappeared}, {appeared}).

    Returns:
        list: A list of UCI moves
    """

    uci_moves = []

    for disappeared, appeared in differences:

        l_d = len(disappeared)
        l_a = len(appeared)

        if l_d == 1 and l_a == 1: # movement or promotion

            old = list(disappeared)[0]
            new = list(appeared)[0]

            if old[:2] == new[:2]: # if the class and color is the same -> movement
                msg = f"{old[2]}{old[3]}{new[2]}{new[3]}"

            else: # promotion
                msg = f"{old[2]}{old[3]}{new[2]}{new[3]}{KEY[new[1]]}"

        elif l_d == 2 and l_a == 1: # capturing

            new = list(appeared)[0]
            old = [piece for piece in disappeared if piece[:2] == new[:2]][0] # the capturer

            msg = f"{old[2]}{old[3]}{new[2]}{new[3]}"

        elif l_d == 2 and l_d == 2: # castling

            new_king = [piece for piece in appeared if piece[:2] == KING][0]

            if new_king[2] == "g": # king side
                if new_king[0] == BLACK:
                    msg = "e8g8"
                else:
                    msg = "e1g1"
            elif new_king[2] == "c": # queen side
                if new_king[0] == BLACK:
                    msg = "e8c8"
                else:
                    msg = "e1c1"
            else:
                msg = "CASTLING_ERROR"

        else:

            msg = "MOVE_ERROR"

        uci_moves.append(msg)
     
    return uci_moves

def generate_pgn(moves, ori):

    """
    Convert UCI moves on a board to PGN and also output board state 

    Parameters:
        moves (list) : List of UCI moves in the game
        ori (list) : List of pieces on the board. Each piece is structured as (color, class, alphabet, num)

    Returns:
        pgn (str) : A string of PGN of the game
        board_states (list) : a list of board states. Each state is a string
    
    """

    # initialize board
    board = chess.Board()

    # if there is a predetermined starting point
    if ori is not None:
        board.clear()

        # place each piece
        for color, piece, alpha, num in ori:
            if color == BLACK:
                piece = piece.lower()
            elif color == WHITE:
                piece = piece.upper()

            position = alpha + str(num)
            square = chess.parse_square(position)
            board.set_piece_at(square, chess.Piece.from_symbol(piece))

    # create game
    game = chess.pgn.Game()
    node = game
    node.headers["FEN"] = board.fen()

    # iterate through UCI and keep track of board state
    board_states = list()
    board_states.append(board)
    for move in moves:
        try:
            node = node.add_variation(board.parse_uci(move))
            board.push_uci(move)
            board_states.append(board)
        except ValueError:
            print(f"Move {move} is invalid")
    
    pgn = str(game).split("\n")[-1]
    return pgn, board_states

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
    uci = uci_from_differences(differences)
    pgn, board_states = generate_pgn(moves=uci, ori=ori)

    return pgn, board_states   


if __name__ == "__main__":
    # Replace with your video file path
    video_path = "2_move_student.mp4"
    pgn, board_states = main(video_path)
    print(pgn)
