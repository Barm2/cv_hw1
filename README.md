# Surgical Tool Object Detection System

This project is an object detection system for identifying surgical tools in images and videos using a YOLOv8 model. The model is trained to detect two surgical Tools: tweezers and Needle driver, and to decet a empty hand. the model was pretrained on a small amount of images, and used semi-supervised learning to improve its performance 

## Installation

To run this project, you need to have Python installed. Follow the steps below to set up the environment and install the required packages.

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/surgical-tool-detection.git
   cd cv_hw1

2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

3. **Install the required packages:**
   ```sh
   pip install -r requirements.txt


## Usage:
This project can be used for surgical tools detection on images and videos:

### Predicting on an Image
The **`predict.py`** script is used for predicting bounding boxes on a single image.

#### Instructions:
1. download **`weights.pt`** from this repository
2. Open the **`predict.py`** file.
3. Modify the paths for the YOLOv8 weights and the input image directly in the code, in the **main** function.
   ```sh
   weights_path = "path/to/your/weights"
   image_path = "path/to/you/image"
4. To create an output image, modify _create_output_image_ to _True_, in the **main** function, as follows:
   ```sh
   predictions = predict_on_image(model, image_path, output_path, create_output_image=True)
5. To generate a predictions file **only** in the (x center, y center, w, h, conf, class) format, modify _create_output_image_ to _False_, in the **main** function, as follows:
   ```sh
   predictions = predict_on_image(model, image_path, output_path, create_output_image=False)
6. The image and prediction file will be created in the project folder

### Predicting on a video
The **`video.py`** script is used for predicting bounding boxes on a video.

#### Instructions:
1. download **`weights.pt`** from this repository
2. Open the **`video.py`** file.
3. Modify the paths for the YOLOv8 weights and the input video directly in the code, in the **main** function.
   ```sh
   weights_path = "path/to/your/weights"
   video_path = "path/to/you/video"
   output_path = "path/to/desired/output/path/for/prediction/files/only"
4. To create an output video, modify _generate_output_video_ to _True_, in the **main** function, as follows:
   ```sh
   predict_on_video(model, video_path, output_path, generate_output_video=True)
5. To generate prediction files **only** in the (x center, y center, w, h, conf, class) format, modify _generate_output_video_ to _False_, in the **main** function, as follows:
   ```sh
   predict_on_video(model, video_path, output_path, generate_output_video=False)
6. The video will be created in the project folder. The prediction files will be created in the desired path, inside **frame_predictions** folder
