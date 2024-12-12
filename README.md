# Competition: Chess Move Tracking

Link to competition on Kaggle : [Chess Detection Competition](https://www.kaggle.com/competitions/cu-chess-detection)

## Motivation & Requirment
- Design program to detect chess piece.
- Use image processing to solve this problem.
- Output Portable Game Notation (PGN) format.
- Can detect chess pieces moving each turn and another side (white- black).
- The algorithm can detect chess on video.
- Visualize its. (Optional) 

# Model Prediction
### Chess Piece Recognition
Users can provide a chess video or image following the guidelines in ./chess_model/evaluation.ipynb to obtain labeled chess pieces.

![image](https://github.com/user-attachments/assets/8520cc6e-d4c4-424b-b097-9ad551833420)

https://github.com/user-attachments/assets/b3fb0e56-b470-4470-adf6-d08f8380c8ea

### Chess Video to PGN 
Users can provide a chess video following the guidelines in AI_boys.ipynb to obtain the PGN file for recording chess games.

| row_id |  output  |
|:-----|:-----|
| 2_Move_rotate_student.mp4   | 1. Ng6 fxg3 * |
| 6_Move_student.mp4   |  1. Qe4= Be1=Q 2. Qe8=B c4 *  |
| 8_Move_student.mp4   | 1. Qe1=K b4= 2. Qe1=K Qe1=K 3. Rxc8 c6=B 4. Be... |

## Project Pipeline

### Preprocessing
- Extract frames from video
- Crop the frame into square aspect ratio
- Tell apart bad frames (hands detected)

### Model Inference
- Feed the preprocessed frame to a YOLO model
- Details on training [here]

### Image processing
- Warp the image and the model's result to a straight square coordinate
- Details about that [here]

### Postprocessing
- Get the cell name for each piece and collect into a board state
- Create a record of meaningful changes between states
- Get PGN from the change log
