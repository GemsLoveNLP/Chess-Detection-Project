# Competition: Chess Move Tracking

Link to competition on Kaggle : [Chess Detection Competition](https://www.kaggle.com/competitions/cu-chess-detection)

## Motivation & Requirment
- Design program to detect chess piece.
- Use image processing to solve this problem.
- Output Portable Game Notation (PGN) format.
- Can detect chess pieces moving each turn and another side (white- black).
- The algorithm can detect chess on video.
- Visualize its. (Optional) 

## Pipeline

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
- Get the cell name for each piece
- Create a record of meaningful changes
- Use those change log to get UCI chess format
- Get PGN and the board's look at each stage using python-chess

## Challenges

### Reading from Video
- Human hands can appear in VDO and cause fail detection
- Glitches could occur for a frame and mess up detection

### Postprocessing
- PGN is not just a function of state differences
- Specificity, Check, Checkmate are all in PGN and is a function of the entire board
- We used UCI instead as it is a function of differences
- chess.pgn.game to convert from UCI to PGN
