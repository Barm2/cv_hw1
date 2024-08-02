from ultralytics import YOLO
import torch
import os
import cv2
import yaml
import random

CLASS_COLORS = {
        0: (255, 255, 255),   # White for class 0
        1: (255, 255, 0),   # Cyan for class 1
        2: (0, 255, 255),     # Yellow for class 2
    }

def predict_and_plot_on_boxes_frame(model, frame):
    """
    predict on image and plot the prediction
    """
    results = model(frame, iou=0.4, conf=0.5, augment=True)

    for i, result in enumerate(results):
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confs = result.boxes.conf  # Confidence scores
        clses = result.boxes.cls   # Class indices

    # Draw bounding boxes on the frame
    for box, conf, cls in zip(boxes, confs, clses):
        x1, y1, x2, y2 = map(int, box)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        color = CLASS_COLORS[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    
    return frame


def predict_on_image(model, image_path, output_path, create_output_image=False):
    """
    predicts bounfing boxes using a givel model.
    outputs an array with the predictions
    the predictions on the image are in format (x center, y center, w, h, conf, class)
    """
    results = model(image_path, iou=0.4, conf=0.5, augment=True)
    result = results[0]
    if create_output_image:
        img = cv2.imread(image_path)
        img = predict_and_plot_on_boxes_frame(model, img)
        image_filename = 'output_image.jpg'
        cv2.imwrite(image_filename, img)
    boxes = result.boxes.xywhn.cpu().numpy()  # Bounding box coordinates
    confs = result.boxes.conf.cpu().numpy()  # Confidence scores
    classes = result.boxes.cls.cpu().numpy()   # Class indices
    predictions = []
    with open(output_path, 'w') as f:
        for box, conf, cls in zip(boxes, confs, classes):
            x_center, y_center, w, h = box
            f.write(f'{x_center} {y_center} {w} {h} {conf} {int(cls)}\n')
            predictions.append((x_center, y_center, w, h, conf, int(cls)))
    return predictions


def main():
    weights_path = "weights/best.pt" #do not chabge
    image_path = "/datashare/HW1/labeled_image_data/images/train/1c0b1584-frame_1789.jpg" #change to your path 
    output_path = "image_label_output.txt"
    model = YOLO(weights_path)
    predictions = predict_on_image(model, image_path, output_path, create_output_image=True)
    for pred in predictions:
        print(pred)
        



if __name__ == '__main__':
    main()
