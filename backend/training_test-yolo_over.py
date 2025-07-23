import os
from ultralytics import YOLO

DATA_YAML_PATH = os.path.join('..', 'dataset', 'data.yaml')  

model = YOLO('yolov8n.pt') 

model.train(
    data=DATA_YAML_PATH,
    epochs=50,
    imgsz=640,
    project='runs',
    name='cana_detect',
    exist_ok=True
)