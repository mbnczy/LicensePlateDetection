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
        detections = self.model.readtext(lp_thresholded)
        for det in detections:
            bbox, text, score = det
            formatted_text = text.upper().replace(' ','')
            return formatted_text,score,lp_thresholded
        lp_Text='error'
        lp_Conf_score = 1
        return lp_Text, lp_Conf_score,lp_thresholded
    
    def Read(self, license_plate):
        #read
        detections = self.model.readtext(license_plate)
        for det in detections:
            bbox, text, score = det
            formatted_text = text.upper().replace(' ','')
            return formatted_text,score,license_plate
        lp_Text='error'
        lp_Conf_score = 1
        return lp_Text, lp_Conf_score,license_plate
    
    def MultiplyPrepAndRead(self,license_plate):
        formatted_text = None
        results_list = []
        lp_gray = cv2.cvtColor(license_plate,cv2.COLOR_BGR2GRAY)
        sharpening_filter = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
        sharp_image = cv2.filter2D(lp_gray, -1, sharpening_filter)
        retval,lp_thresholded = cv2.threshold(sharp_image,34,255,cv2.THRESH_BINARY_INV)

        detections = self.model.readtext(lp_thresholded)
        for det in detections:
            bbox, text, score = det
            formatted_text = text.upper().replace(' ','')
        if formatted_text is not None and self.CheckSyntax(formatted_text):
            results_list.append((formatted_text,score))

        inv_thresholded = cv2.bitwise_not(lp_thresholded)
        
        detections = self.model.readtext(inv_thresholded)
        for det in detections:
            bbox, text, score = det
            formatted_text = text.upper().replace(' ','')
        if formatted_text is not None and self.CheckSyntax(formatted_text):
            results_list.append((formatted_text,score))


        median_filtered = cv2.medianBlur(inv_thresholded, 3)
        gauss3_filtered = cv2.GaussianBlur(inv_thresholded, (3,3), 2)
        gauss2_filtered = cv2.GaussianBlur(gauss3_filtered, (3,3), 2)
        gauss_filtered = cv2.GaussianBlur(gauss2_filtered, (3,3), 2)

        detections = self.model.readtext(gauss_filtered)
        for det in detections:
            bbox, text, score = det
            formatted_text = text.upper().replace(' ','')
        if formatted_text is not None and self.CheckSyntax(formatted_text):
            results_list.append((formatted_text,score))
        try:
            formatted_text = max(results_list, key=lambda x: x[1])[0]
            score = max(results_list, key=lambda x: x[1])[1]
            # Additional code if needed
        except ValueError:
            print("Error: results_list is empty or contains tuples without a second element.")
            formatted_text = None
            score = None
            # Handle the exception as needed

        return formatted_text, score, gauss_filtered
        

    def ModifiedPrepAndRead(self,license_plate):
        sharpening_filter = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
        lp_gray = cv2.cvtColor(license_plate,cv2.COLOR_BGR2GRAY)
        sharp_image = cv2.filter2D(lp_gray, -1, sharpening_filter)
        #sharp_image = cv2.filter2D(sharp_image, -1, sharpening_filter)
        retval,lp_thresholded = cv2.threshold(sharp_image,34,255,cv2.THRESH_BINARY_INV)
        inv_thresholded = cv2.bitwise_not(lp_thresholded)
        median_filtered = cv2.medianBlur(inv_thresholded, 3)
        gauss3_filtered = cv2.GaussianBlur(inv_thresholded, (3,3), 2)
        gauss2_filtered = cv2.GaussianBlur(gauss3_filtered, (3,3), 2)
        gauss_filtered = cv2.GaussianBlur(gauss2_filtered, (3,3), 2)
        final = cv2.filter2D(gauss_filtered, -1, sharpening_filter) 

        

        detections = self.model.readtext(final)
        for det in detections:
            bbox, text, score = det
            formatted_text = text.upper().replace(' ','')
            if self.CheckSyntax(formatted_text):
                return formatted_text,score,license_plate
        lp_Text='error: no det'
        lp_Conf_score = 0
        return lp_Text, lp_Conf_score,license_plate

    def CheckSyntax(self, text: str) ->bool:
        return ((not any(not c.isalnum() for c in text)) and len(text) > 5)


 