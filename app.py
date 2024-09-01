from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from PIL import Image
import io
import math
import torch
import torchvision.ops as ops

app = FastAPI()

class FishMedakaAI:
    def __init__(self, model_paths=None, alphas=None):  # Corrected from _init_ to __init__
        if model_paths:
            self.models = [YOLO(path) for path in model_paths]
            self.alphas = alphas
            self.detector_model_path = model_paths[0]  # Using the first model as the detector
        else:
            self.detector_model_path = 'yolov8_best_medaka_detector.pt'

        self.class_names = ['O.celebensis', 'O.javanicus']

    def detect_medaka_experiment1_single_model(self, image: np.ndarray):
        detector_model = YOLO(self.detector_model_path)
        iou_threshold = 0.4
        confidence_score_threshold = 0.6

        results = detector_model([image], iou=iou_threshold, conf=confidence_score_threshold)

        detection_data = {
            'image_base64': '',
            'image_processed_base64_list': [],
            'classes': [],
            'confidence_scores': []
        }

        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped_images = []

        for result in results:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, confidence, class_id = box
                class_name = self.class_names[int(class_id)]

                detection_data['classes'].append(class_name)
                detection_data['confidence_scores'].append(float(confidence))

                cropped_img = original_image[int(y1):int(y2), int(x1):int(x2)]
                cropped_images.append((cropped_img, class_name, confidence))

            im_bgr = result.plot()
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(im_rgb)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            detection_data['image_base64'] = encoded_image

        for crop, class_name, confidence in cropped_images:
            if crop.size != 0:
                pil_crop_image = Image.fromarray(crop)
                buffered = io.BytesIO()
                pil_crop_image.save(buffered, format="JPEG")
                encoded_crop_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                detection_data['image_processed_base64_list'].append(encoded_crop_image)

        return detection_data

    def ensemble_adaboost(self, image: np.ndarray):
        all_boxes = []
        all_scores = []
        all_classes = []

        for model, alpha in zip(self.models, self.alphas):
            results = model([image], conf=0.25)

            for result in results:
                boxes = result.boxes
                all_boxes.append(boxes.xyxy)
                all_scores.append(boxes.conf * alpha)
                all_classes.append(boxes.cls)

        if all_boxes:
            boxes_tensor = torch.cat(all_boxes)
            scores_tensor = torch.cat(all_scores)
            classes_tensor = torch.cat(all_classes)

            nms_indices = ops.nms(boxes_tensor, scores_tensor, 0.5)

            final_boxes = boxes_tensor[nms_indices]
            final_scores = scores_tensor[nms_indices]
            final_classes = classes_tensor[nms_indices]

            return final_boxes, final_scores, final_classes

        return [], [], []

    def detect_medaka_experiment3_ensemble_model(self, image: np.ndarray):
        final_boxes, final_scores, final_classes = self.ensemble_adaboost(image)

        detection_data = {
            'image_base64': '',
            'image_processed_base64_list': [],
            'classes': [],
            'confidence_scores': []
        }

        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_copy = original_image.copy()  # Create a copy of the original image for drawing

        for box, score, class_id in zip(final_boxes, final_scores, final_classes):
            x1, y1, x2, y2 = box.cpu().numpy()
            confidence = score.item()
            class_name = self.class_names[int(class_id)]

            detection_data['classes'].append(class_name)
            detection_data['confidence_scores'].append(float(confidence))

            cropped_img = original_image[int(y1):int(y2), int(x1):int(x2)]

            if cropped_img.size != 0:
                pil_crop_image = Image.fromarray(cropped_img)
                buffered = io.BytesIO()
                pil_crop_image.save(buffered, format="JPEG")
                encoded_crop_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                detection_data['image_processed_base64_list'].append(encoded_crop_image)

            # Draw the bounding boxes on the image copy
            cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(image_copy, f"{class_name} {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert the modified image with detections back to base64
        pil_image = Image.fromarray(image_copy)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        detection_data['image_base64'] = encoded_image

        return detection_data

# Initialize FishMedakaAI with default paths for the ensemble model
model_paths = ['best_model1.pt', 'best_model2.pt', 'best_model3.pt', 'best_model4.pt', 'best_model5.pt']
alphas = [0.5 * math.log((1 - rate) / rate) for rate in [0.2, 0.2, 0.2, 0.2, 0.2]]
fish_ai_ensemble = FishMedakaAI(model_paths, alphas)

@app.post("/detect_medaka_experiment1/")
async def detect_medaka_experiment1(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        fish_ai_single = FishMedakaAI()
        result = fish_ai_single.detect_medaka_experiment1_single_model(image)

        return JSONResponse(content={"detection_results": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_medaka_experiment3_ensemble/")
async def detect_medaka_experiment3_ensemble(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        result = fish_ai_ensemble.detect_medaka_experiment3_ensemble_model(image)

        return JSONResponse(content={"detection_results": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":  # Corrected from _name_ to __name__
    uvicorn.run(app, host="0.0.0.0", port=8000)
