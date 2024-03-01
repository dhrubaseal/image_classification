from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data=r'C:\Personal_Projects\projects\Image Classification\image_classification\data', epochs=1, imgsz=64)