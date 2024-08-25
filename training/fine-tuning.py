# Description: This script downloads the dataset from Roboflow and fine-tunes the YOLOv5 model on it. Use it in Jupiter Notebook with GPU.
from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the Roboflow client
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
# Download the dataset
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
# Fine-tune on YOLOv5
dataset = version.download("yolov5")

# Move the dataset to the correct directory while in Jupyter Notebook
import shutil
shutil.move('football-players-detection-1/train', 'football-players-detection-1/football-players-detection-1/train')
shutil.move('football-players-detection-1/test', 'football-players-detection-1/football-players-detection-1/test')
shutil.move('football-players-detection-1/valid', 'football-players-detection-1/football-players-detection-1/valid')

# Model fine-tuning (use this command in the terminal or Jupyter Notebook) 
#!yolo task=detect mode=train model=yolov5l.pt data={dataset.location}/data.yaml epochs=100 imgsz=640      