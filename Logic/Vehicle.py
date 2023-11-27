from ultralytics import YOLO
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class VehicleDetection():
    def __init__(self,modeltype: str) -> None:
        self.type = modeltype
        if modeltype == 'YOLOv8':
            self.model = YOLO('yolov8n.pt')
            self.vehicle_ids = [2,3,5,7]
    def Detect(self, frame):
        return self.model(frame)[0]


