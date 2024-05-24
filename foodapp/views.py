# views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .yolov7.inference import ImagePredictor

class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            # 從請求中獲取上傳的圖片文件
            image = request.FILES.get('image')

            # 確保有圖片文件被上傳
            if image is None:
                raise ValueError("未上傳圖片。")

            # 初始化影像預測器，指定模型路徑和設備
            model_path = "C:/Users/USER/foodproject/foodproject/static/models/New26L__0519_lr00012_850e.pt"
            predictor = ImagePredictor(model_path, device='cpu')

            # 使用影像預測器進行圖像辨識，得到辨識結果和圖片尺寸
            prediction, img_size = predictor.predict_image(image)

            # 檢查辨識結果是否為空
            if prediction is None:
                raise ValueError("圖片辨識失敗。")

            # 將辨識結果轉換為字典格式
            result = predictor.write_prediction_to_txt(prediction)

            # 回傳辨識結果，HTTP狀態碼為200表示成功
            return Response({'prediction': result}, status=status.HTTP_200_OK)
        except Exception as e:
            # 捕獲並處理可能的異常情況，回傳錯誤訊息，HTTP狀態碼為400表示客戶端錯誤
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
