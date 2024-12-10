from ultralytics.utils.ops import xyxy2xywh

class ChessPredicter:
    def __init__(self, model):
      self.model = model

    def predict_image(self, img):
      results = self.model(img)[0]
      return results.to_json()

    def predict_video(self):
      pass
    
    def get_base_coordinates(self, xywh):
      pass