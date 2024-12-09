from ultralytics.utils.ops import xyxy2xywh

class ChessPredicter:
    def __init__(self, model):
      self.model = model

    def predict_image(self, img_path):
      results = self.model(img_path)[0]
      return results.to_json()

    def predict_video(self):
      pass
    
    def get_base_coordinates(self, xywh):
      pass