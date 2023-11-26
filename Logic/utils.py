from ultralytics import YOLO
import cv2
import pandas as pd
from RealtimeTracking import *
import gc
import easyocr
import numpy as np
import csv
from scipy.interpolate import interp1d



def PlateToVehicle(det_license_plate,det_vehicles):
    lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = det_license_plate
    found=False
    for i in range(len(det_vehicles)):
        v_x1, v_y1, v_x2, v_y2, v_id = det_vehicles[i]
        #inside
        if lp_x1 > v_x1 and lp_y1 > v_y1 and lp_x2 < v_x2 and lp_y2 < v_y2:
            v_index=i
            found=True
            break
    if found:
        return det_vehicles[v_index]
    return -1,-1,-1,-1,-1

def interpolate_bounding_boxes(data, ShowOnlyBestScore: bool):
    if ShowOnlyBestScore:
        return interpolate_bounding_boxes_bestScore(data)
    else:
        return interpolate_bounding_boxes_all(data)

def interpolate_bounding_boxes_all(data):

    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['fr_number']) for row in data])
    car_ids = np.array([int(float(row['v_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['v_bbox'][1:-1].split(', '))) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['lp_bbox'][1:-1].split(', '))) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['fr_number'] for p in data if int(float(p['v_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['fr_number'] = str(frame_number)
            row['v_id'] = str(car_id)
            row['v_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['lp_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['lp_bbox_score'] = '0'
                row['lp'] = '0'
                row['lp_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['fr_number']) == frame_number and int(float(p['v_id'])) == int(float(car_id))][0]
                row['lp_bbox_score'] = original_row['lp_bbox_score'] if 'lp_bbox_score' in original_row else '0'
                row['lp'] = original_row['lp'] if 'lp' in original_row else '0'
                row['lp_score'] = original_row['lp_score'] if 'lp_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data

def interpolate_bounding_boxes_bestScore(data):

    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['fr_number']) for row in data])
    car_ids = np.array([int(float(row['v_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['v_bbox'][1:-1].split(', '))) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['lp_bbox'][1:-1].split(', '))) for row in data])
    lp_confidences = np.array([float(row.get('lp_score', '0')) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []
        best_confidence_idx = np.argmax(lp_confidences[car_mask])
        best_confidence_lp = license_plate_bboxes[car_mask][best_confidence_idx]
        

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['fr_number'] = str(frame_number)
            row['v_id'] = str(car_id)
            row['v_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['lp_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            # Include license plate information with the best confidence score
            best_confidence_indices = np.where(car_mask)[0]
            best_confidence_row_idx = best_confidence_indices[best_confidence_idx]
            
            #row['lp_bbox'] = ' '.join(map(str, best_confidence_lp))
            row['lp_bbox_score'] = str(lp_confidences[best_confidence_row_idx])
            row['lp'] = data[best_confidence_row_idx]['lp']
            row['lp_score'] = data[best_confidence_row_idx]['lp_score']
            interpolated_data.append(row)

    return interpolated_data

def interpolate_license_plates(data,bestscore=True):
    if bestscore:
        return modified_interpolate(data)
    else:
        return modified_interpolate_all(data)
    
def modified_interpolate(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['fr_number']) for row in data])
    ids = np.array([int(float(row['id'])) for row in data])
    #car_bboxes = np.array([list(map(float, row['v_bbox'][1:-1].split(', '))) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['lp_bbox'][1:-1].split(', '))) for row in data])
    lp_confidences = np.array([float(row.get('lp_score', '0')) for row in data])

    interpolated_data = []
    unique_ids = np.unique(ids)

    for id in unique_ids:
        # Filter data for a specific car ID
        mask = ids == id
        lp_frame_numbers = frame_numbers[mask]
        license_plate_bboxes_interpolated = []
        best_confidence_idx = np.argmax(lp_confidences[mask])
        best_confidence_lp = license_plate_bboxes[mask][best_confidence_idx]
        

        first_frame_number = lp_frame_numbers[0]
        last_frame_number = lp_frame_numbers[-1]

        for i in range(len(license_plate_bboxes[mask])):
            frame_number = lp_frame_numbers[i]
            bbox = license_plate_bboxes[mask][i]
            license_plate_bbox = license_plate_bboxes[mask][i]

            if i > 0:
                prev_frame_number = lp_frame_numbers[i-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(license_plate_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['fr_number'] = str(frame_number)
            row['id'] = str(id)
            row['lp_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            # Include license plate information with the best confidence score
            best_confidence_indices = np.where(mask)[0]
            best_confidence_row_idx = best_confidence_indices[best_confidence_idx]
            
            #row['lp_bbox'] = ' '.join(map(str, best_confidence_lp))
            row['lp_bbox_score'] = str(lp_confidences[best_confidence_row_idx])
            row['lp'] = data[best_confidence_row_idx]['lp']
            row['lp_score'] = data[best_confidence_row_idx]['lp_score']
            interpolated_data.append(row)

    return interpolated_data

def modified_interpolate_all(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['fr_number']) for row in data])
    ids = np.array([int(float(row['id'])) for row in data])
    #car_bboxes = np.array([list(map(float, row['v_bbox'][1:-1].split(', '))) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['lp_bbox'][1:-1].split(', '))) for row in data])
    lp_confidences = np.array([float(row.get('lp_score', '0')) for row in data])

    interpolated_data = []
    unique_ids = np.unique(ids)

    for id in unique_ids:
        # Filter data for a specific car ID
        mask = ids == id
        lp_frame_numbers = frame_numbers[mask]
        license_plate_bboxes_interpolated = []
        frame_numbers_ = [p['fr_number'] for p in data if int(float(p['id'])) == int(float(id))]
        

        first_frame_number = lp_frame_numbers[0]
        last_frame_number = lp_frame_numbers[-1]

        for i in range(len(license_plate_bboxes[mask])):
            frame_number = lp_frame_numbers[i]
            bbox = license_plate_bboxes[mask][i]
            license_plate_bbox = license_plate_bboxes[mask][i]

            if i > 0:
                prev_frame_number = lp_frame_numbers[i-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(license_plate_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['fr_number'] = str(frame_number)
            row['id'] = str(id)
            row['lp_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            # Include license plate information with the best confidence score
            #best_confidence_indices = np.where(mask)[0]
            #best_confidence_row_idx = best_confidence_indices[best_confidence_idx]
            
            #row['lp_bbox'] = ' '.join(map(str, best_confidence_lp))
            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['lp_bbox_score'] = '0'
                row['lp'] = '0'
                row['lp_score'] = '0'
            else:
                original_row = [p for p in data if int(p['fr_number']) == frame_number and int(float(p['id'])) == int(float(id))][0]
                row['lp_bbox_score'] = original_row['lp_bbox_score'] if 'lp_bbox_score' in original_row else '0'
                row['lp'] = original_row['lp'] if 'lp' in original_row else '0'
                row['lp_score'] = original_row['lp_score'] if 'lp_score' in original_row else '0'
            interpolated_data.append(row)

    return interpolated_data
