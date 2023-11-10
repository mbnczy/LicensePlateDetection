from ultralytics import YOLO
import cv2
import pandas as pd
from RealtimeTracking import *
import gc
import easyocr
import numpy as np
import csv
from scipy.interpolate import interp1d

class LicensePlateDetection():
    def __init__(self,modeltype: str) -> None:
        self.type = modeltype
        if modeltype == 'yolov8n_90e_cust':
            self.model = YOLO('/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/models/YOLOv8/yolov8n_90e_cust/runs/detect/train4/weights/best.pt')
    def Detect(self, frame):
        return self.model(frame)[0]

class LicensePlateReader():
    def __init__(self,modeltype: str) -> None:
        self.type = modeltype
        if modeltype == 'easyocr':
            self.model = easyocr.Reader(['en'],gpu=False)
    def PrepAndRead(self, license_plate):
        #prep
        lp_gray = cv2.cvtColor(license_plate,cv2.COLOR_BGR2GRAY)
        retval,lp_thresholded = cv2.threshold(lp_gray,64,255,cv2.THRESH_BINARY_INV)
        #read
        detections = self.model.readtext(license_plate)
        for det in detections:
            bbox, text, score = det
            formatted_text = text.upper().replace(' ','')
            return formatted_text,score
        lp_Text='error'
        lp_Conf_score = 1
        return lp_Text, lp_Conf_score
 