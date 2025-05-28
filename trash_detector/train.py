from ultralytics import YOLO

model = YOLO("yolov8m.pt")
train_config_file = "train_config.yaml"

model.train(data="train_config.yaml", epochs=100, plots=True)
