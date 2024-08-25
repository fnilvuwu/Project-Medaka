from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from PIL import Image
import io
import json

app = FastAPI()

class FishMedakaAI:
    def __init__(self):
        self.detector_model_path = 'yolov8_best_medaka_detector_v2.pt'
        self.class_names = ['O.celebensis', 'O.javanicus']  # Replace with actual class names

    def detect_medaka(self, image: np.ndarray):
        detector_model = YOLO(self.detector_model_path)
        iou_threshold = 0.4
        confidence_score_threshold = 0.6
        
        results = detector_model([image], iou=iou_threshold, conf=confidence_score_threshold)
        
        detection_data = {
            'image_base64': '',
            'image_processed_base64_list': [],  # List to store multiple cropped images
            'classes': [],
            'confidence_scores': []
        }

        # Convert BGR image to RGB for processing
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cropped_images = []

        for result in results:
            # Extract classes, confidence scores, and crop images from result
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, confidence, class_id = box
                class_name = self.class_names[int(class_id)]
                
                detection_data['classes'].append(class_name)
                detection_data['confidence_scores'].append(float(confidence))
                
                # Crop the detected object
                cropped_img = original_image[int(y1):int(y2), int(x1):int(x2)]
                cropped_images.append((cropped_img, class_name, confidence))

            # Plot the results image with bounding boxes
            im_bgr = result.plot()  # BGR-order numpy array
            
            # Convert BGR to RGB format
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert the image to PIL format and then to base64
            pil_image = Image.fromarray(im_rgb)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            detection_data['image_base64'] = encoded_image

        # Convert each cropped image to base64 format and store in list
        for crop, class_name, confidence in cropped_images:
            if crop.size != 0:
                # Convert cropped image to PIL format
                pil_crop_image = Image.fromarray(crop)
                buffered = io.BytesIO()
                pil_crop_image.save(buffered, format="JPEG")
                encoded_crop_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                detection_data['image_processed_base64_list'].append(encoded_crop_image)

        # Return the detection data
        return detection_data

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
        
        return JSONResponse(content={"detection_results": result})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
