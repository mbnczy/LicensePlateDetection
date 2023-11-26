from ultralytics import YOLO
import cv2
import pandas as pd
from RealtimeTracking import *
import gc
import numpy as np 
import csv
from scipy.interpolate import interp1d
import os
from tqdm import tqdm
from LicensePlate import LicensePlateDetection, LicensePlateReader
from Vehicle import VehicleDetection
from utils import interpolate_license_plates
import ast
from datetime import datetime, timedelta

# %%
gc.collect()
# %%
plate_det_model_path = '/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/models/YOLOv8/yolov8n_90e_cust/runs/detect/train4/weights/best.pt'
testvideo_path = '/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/video_data/IMG_0493.mp4'
ShowOnlyBestScore = True

# %%
#visualization
def draw_bounding_boxes_lp(input_video_path, interpolated_df, output_video_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(interpolated_df)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file: {input_video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # You can change the codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame and draw bounding boxes
    frame_number = 0
    ret = True
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Add tqdm to track progress
    for _ in tqdm(range(total_frames), desc="Write on frames", unit="frame"):
        ret, frame = cap.read()

        # Filter the DataFrame for the current frame number
        frame_data = df[df['fr_number'] == frame_number]        

        for _, row in frame_data.iterrows():
            lp_bbox_vals = row['lp_bbox'].split(' ')
            lp_bbox_vals[:]=map(lambda x: int(float(x)),lp_bbox_vals)
            lp_bbox_vals = str(lp_bbox_vals)
            lp_bbox_vals_str = "[" + lp_bbox_vals[1:-1] + "]"
            license_plate_bbox = ast.literal_eval(lp_bbox_vals_str)

            H = license_plate_bbox[3] - license_plate_bbox[1]
            W = license_plate_bbox[2] - license_plate_bbox[0]

            # Draw bounding boxes on the frame
            cv2.rectangle(frame, (int(license_plate_bbox[0]), int(license_plate_bbox[1])),
                          (int(license_plate_bbox[2]), int(license_plate_bbox[3])), (0, 0, 255), 2)

            # Show license plate text above the license plate bounding box
            if str(row['lp']) != '0':
                (text_width, text_height), _ = cv2.getTextSize(
                    str(row['lp']),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    5)
                text_x = int((license_plate_bbox[0] + license_plate_bbox[2] - text_width) / 2)
                text_y = int(license_plate_bbox[1] - text_height - 5)  # Adjusted position
                #cv2.rectangle(frame, (license_plate_bbox[0]-text_width+W, license_plate_bbox[1]-text_height-H), (license_plate_bbox[2]+text_width-W, license_plate_bbox[1]), (0, 0, 0), thickness=cv2.FILLED)
                #cv2.rectangle(frame, (int(license_plate_bbox[2]-W), int(license_plate_bbox[3]-H-H)),
                #          (int(license_plate_bbox[0])+W, int(license_plate_bbox[1])), (0, 0, 0), thickness=cv2.FILLED)
                cv2.rectangle(frame, (text_x-5, text_y+5-text_height*2),
                          (text_x+text_width+5, text_y+5), (0, 0, 0), thickness=cv2.FILLED)
                
                cv2.putText(frame,
                            str(row['lp']),
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255, 255, 255),
                            3)

        # Write the frame with bounding boxes to the output video
        if frame is not None:
            out.write(frame)

        frame_number += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()


def WriteLog(df: pd.DataFrame, path: str):
    u_ids = np.unique(df['id'])
    times = []
    for id in u_ids:
        newdf = df[df['id']==id].sort_values(by='fr_number')
        times.append({'id': id,
                     'license_plate': df.loc[df[df['id']==id]['lp_score'].idxmax()]['lp'],
                     'confidence_score':df[df['id']==id]['lp_score'].max(),
                     'arrive': (creation_datetime+timedelta(seconds=int(newdf.head(1)['fr_number'])*spf)).strftime("%Y.%d.%m. %H:%M:%S"), 
                     'leave': (creation_datetime+timedelta(seconds=int(newdf.tail(1)['fr_number'])*spf)).strftime("%Y.%d.%m. %H:%M:%S")})
    pd.DataFrame(times, columns=['id','license_plate','confidence_score','arrive','leave']).sort_values(by='arrive').reset_index().drop('index',axis=1).to_csv(path)


# OOP
df_cols = ['fr_number','id', 'lp_bbox','lp_bbox_score','lp','lp_score']
df_rows = []

video_capture = cv2.VideoCapture(testvideo_path)
fps = video_capture.get(cv2.CAP_PROP_FPS)
spf = 1/fps
creation_datetime = datetime.fromtimestamp(os.stat(testvideo_path).st_birthtime)

lp_detection = LicensePlateDetection('yolov8n_90e_cust')
lp_reader = LicensePlateReader('easyocr')

ret = True
frame_indexer = 0
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

for _ in tqdm(range(total_frames), desc="Detecting license plates", unit="frame"):
    if not ret or video_capture.isOpened() == False:
        break

    ret, frame = video_capture.read()

    if ret:
        det_lps = lp_detection.Detect(frame)
        platesonframe = []

        for lp_index, det_lp in enumerate(det_lps.boxes.data.tolist()):
            lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = det_lp

            platesonframe.append(det_lp)
            if LicensePlateDetection.noOverlap(platesonframe):
                license_plate = frame[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2), :]
                lp_text, lp_text_confscore, lp_prep = lp_reader.ModifiedPrepAndRead(license_plate)
                if lp_text != None:
                    df_rows.append({'fr_number': frame_indexer,
                                    'id': lp_index,
                                    'lp_bbox': [lp_x1, lp_y1, lp_x2, lp_y2],
                                    'lp_bbox_score': lp_score,
                                    'lp': lp_text,
                                    'lp_score': lp_text_confscore})
    frame_indexer += 1

out = pd.DataFrame(df_rows,columns=df_cols)

WriteLog(out,'/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/time_log.csv')

gc.collect()

out = out.to_dict('records')

interpolateable = []
for i,row in enumerate(out):
    interpolateable.append({'': str(i),
                   'fr_number': str(row.get('fr_number', '')),
                   'id': str(row.get('id', '')),
                   'lp_bbox': str(row.get('lp_bbox', '')),
                   'lp_bbox_score': str(row.get('lp_bbox_score', '')),
                   'lp': str(row.get('lp', '')),
                   'lp_score': str(row.get('lp_score', ''))})
gc.collect()

# %%
# Interpolate missing data
interpolated_data = interpolate_license_plates(interpolateable,ShowOnlyBestScore)

# %%
gc.collect()


# Write updated data to a new CSV file
header = ['fr_number','id', 'lp_bbox','lp_bbox_score','lp','lp_score']
with open('/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/log_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

gc.collect()

# %%
draw_bounding_boxes_lp(testvideo_path,
                    '/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/log_interpolated.csv',
                    testvideo_path[0:-4]+'lp.avi')