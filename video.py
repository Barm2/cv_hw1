from ultralytics import YOLO
import torch
import os
import cv2
import yaml
import random
from predict import predict_on_image, predict_and_plot_on_boxes_frame




def create_new_folder_for_predictions(output_path):
    new_folder_path = os.path.join(output_path, 'frame_predictions')
    os.makedirs(new_folder_path, exist_ok=True)
    return new_folder_path


def create_only_predictions_on_video(model, video_path, output_path):
    prediction_folder_path = create_new_folder_for_predictions(output_path)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prediction_path = os.path.join(prediction_folder_path, f'vid_frame_{frame_count:06d}.txt')
        predictions = predict_on_image(model, frame, prediction_path)
        frame_count += 1

    
def create_output_video(model, video_path, output_path, output_name='output_video.mp4'):
    prediction_folder_path = create_new_folder_for_predictions(output_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        prediction_path = os.path.join(prediction_folder_path, f'vid_frame_{frame_count:06d}.txt')
        predictions = predict_on_image(model, frame, prediction_path)

        frame = predict_and_plot_on_boxes_frame(model, frame)
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def predict_on_video(model, video_path, output_path, generate_output_video):
    if generate_output_video:
        create_output_video(model, video_path, output_path)
    else:
        create_only_predictions_on_video(model, video_path, output_path)

def main():
    weights_path = "runs/detect/train_with_conf_0.7/weights/best.pt"
    video_path = "/datashare/HW1/ood_video_data/surg_1.mp4"
    output_path = "./"
    model = YOLO(weights_path)
    predict_on_video(model, video_path, output_path, generate_output_video=True)

        

if __name__ == '__main__':
    main()