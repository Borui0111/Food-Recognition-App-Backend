import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import cv2
from PIL import Image
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import io

class ImagePredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = select_device(device)
        self.model = attempt_load(model_path, map_location=self.device).eval()
        self.model_path = model_path  

    def predict_image(self, image):
        try:
            img0 = Image.open(image).convert("RGB")
        except OSError as e:
            return None, (0, 0)

        # 圖像預處理步驟，例如降噪、銳化、對比度增強等
        img = cv2.cvtColor(np.array(img0), cv2.COLOR_RGB2BGR)
        # 調整圖片大小
        img = cv2.resize(img, (640, 640))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).to(self.device).float()
        img = img.permute(0, 3, 1, 2)
        img = img / 255.0

        with torch.no_grad():
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, 0.2, 0.4)

        return [pred], img0.size[::-1] 

    def write_prediction_to_txt(self, prediction):
        best_prediction = {'食物類別': '無法辨識', '信心度': 0}  

        if prediction[0] is not None and len(prediction[0]):
            highest_confidence = 0.0
            for det in prediction[0]:
                for *xyxy, conf, cls in det:
                    if conf > highest_confidence:
                        highest_confidence = conf
                        best_prediction = {
                            '食物類別': int(cls),
                            '信心度': round(float(conf), 2)
                        }

        return best_prediction
