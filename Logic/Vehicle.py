from ultralytics import YOLO
import cv2
import pandas as pd
from RealtimeTracking import *
import gc
import easyocr
import numpy as np
import csv
from scipy.interpolate import interp1d

class VehicleDetection():
    def __init__(self,modeltype: str) -> None:
        self.type = modeltype
        if modeltype == 'YOLOv8':
            self.model = YOLO('yolov8n.pt')
            self.vehicle_ids = [2,3,5,7]
    def Detect(self, frame):
        return self.model(frame)[0]


