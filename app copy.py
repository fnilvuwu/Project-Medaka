from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import os
from ultralytics import YOLO

app = FastAPI()

class FishMedakaAI:
    def __init__(self):
        self.detector_model_path = 'yolov8_best_medaka_detector.pt'

    def detect_medaka(self, image: np.ndarray):
        detector_model = YOLO(self.detector_model_path)
        iou_threshold = 0.5
        
        results = detector_model([image], iou=iou_threshold)
        
        detection_boxes = []
        detection_probs = []

        for result in results:
            detection_boxes.append(result.boxes)
            detection_probs.append(result.probs)

        return {
            'boxes': detection_boxes,
            'probs': detection_probs
        }

@app.post("/detect/")
async def detect_medaka(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        fish_ai = FishMedakaAI()
        result = fish_ai.detect_medaka(image)
        print(result)
        return result
        # return JSONResponse(content={"detection_results": result})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
