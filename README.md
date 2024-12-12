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

https://github.com/user-attachments/assets/1e722537-152a-4dad-bf01-c932e259486d

### Chess Video to PGN 
Users can provide a chess video following the guidelines in AI_boys.ipynb to obtain the PGN file for recording chess games.

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
