import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from dotenv import dotenv_values
from ultralytics.yolo.utils.ops import scale_image

config = dotenv_values("./.env") 

class ChessPredicter:
    def __init__(self, model):
      self.model_path = model

    def predict_image(self, img):
      result = model(img)[0]

      # detection
      # result.boxes.xyxy   # box with xyxy format, (N, 4)
      cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
      probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
      boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)

      # segmentation
      masks = result.masks.masks.cpu().numpy()     # masks, (N, H, W)
      masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
      # rescale masks to original image
      masks = scale_image(masks.shape[:2], masks, result.masks.orig_shape)
      masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

      return boxes, masks, cls, probs

    def predict_video(self):
      pass
    
    def get_base_coordinates(self, xywh):
      print(xywh[0])
      return xywh[0]
       

print(config.values())

model = YOLO(config["PROJECT_PATH"] + "/chess_model/runs/detect/chess_data_model/weights/best.pt") # path to .pt model file

results = model("./test_data/2_move_student.mp4", stream=True, save=True, show=True, project='./result',)