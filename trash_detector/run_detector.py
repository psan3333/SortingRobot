from ultralytics import YOLO
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("./runs/detect/train5/weights/best.pt")
image = "./data/images/000011.jpg"
result = model(image)
result[0].save("detection_result.jpg")
