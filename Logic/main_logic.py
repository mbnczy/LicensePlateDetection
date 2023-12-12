from ultralytics import YOLO
import cv2
import pandas as pd
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
import ffmpeg
#from RealtimeTracking import Sort
from sort.sort import *

def GetModel(select_name):
    keys = ["Yolo v8 - M (120)",
            "Yolo v8 - N (120)",
            "Yolo v8 - N (90)",
            "Yolo v8 - N (60)",
            "Yolo v8 - N (30)"]
    values = ["yolov8m_120e",
            "yolov8n_120e",
            "yolov8n_90e",
            "yolov8n_60e",
            "yolov8n_30e"]
    return dict(zip(keys, values))[select_name]

def Run(input_path, model, showonlybestconf, im):
    mot_tracker= Sort()

    norm_model_name = GetModel(model)
    #testvideo_path = '/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/video_data/IMG_0493.mp4'
    testvideo_path = input_path
    #print(testvideo_path)
    ShowOnlyBestScore = showonlybestconf

    df_cols = ['fr_number','id', 'lp_bbox','lp_bbox_score','lp','lp_score']
    df_rows = []

    video_capture = cv2.VideoCapture(testvideo_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 60
    
    if type(testvideo_path) is not str:
        creation_datetime = datetime.now
    else:
        creation_datetime = datetime.fromtimestamp(os.stat(testvideo_path).st_birthtime)

    lp_detection = LicensePlateDetection(norm_model_name)
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
            detections = det_lps.boxes.data.tolist()
            tracks = []
            for det in detections:
                tracks.append([det[0],det[1],det[2],det[3],det[4]])
            platesonframe = []
            if len(tracks) != 0:
                track_bbs_ids = mot_tracker.update(np.asarray(tracks))
                
    
                for lp_index, det_lp in enumerate(detections):
                    lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = det_lp
                    if LicensePlateDetection.noOverlapp(platesonframe,det_lp):
                        license_plate = frame[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2), :]
                        if im == 'weak':
                            lp_text, lp_text_confscore, lp_prep = lp_reader.PrepAndRead(license_plate)
                        elif im == 'mid':
                            lp_text, lp_text_confscore, lp_prep = lp_reader.ModifiedPrepAndRead(license_plate)
                        else:
                            lp_text, lp_text_confscore, lp_prep = lp_reader.MultiplyPrepAndRead(license_plate)
                        try:
                            #print(f'good: {track_bbs_ids[lp_index]}, {lp_index}')
                            if lp_index < len(detections):
                                if lp_text != None:
                                    df_rows.append({'fr_number': frame_indexer,
                                                'id': track_bbs_ids[lp_index,4],
                                                'lp_bbox': [lp_x1, lp_y1, lp_x2, lp_y2],
                                                'lp_bbox_score': lp_score,
                                                'lp': lp_text,
                                                'lp_score': lp_text_confscore})
                                    platesonframe.append(det_lp)
                        except:
                            print(f'error: {track_bbs_ids}, {lp_index}')
                            
                    
        frame_indexer += 1
    video_capture.release()
    
    out = pd.DataFrame(df_rows,columns=df_cols)
    out.to_csv('/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/main_logic_log.csv')

    WriteLog(out,'/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/time_log.csv',creation_datetime,fps)

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


    #only for test
    #with open('/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/log_Shortest.csv', 'r') as file:
    #    reader = csv.DictReader(file)
    #    interpolateable = list(reader)    

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

    out_path = str(testvideo_path[0:-4]+'_lp_'+datetime.now().strftime("%Y-%d-%m-%H-%M-%S")+'.mp4')
    # %%
    draw_bounding_boxes_lp(testvideo_path,
                        '/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/log_interpolated.csv',
                        out_path)
    return out_path

def draw_bounding_boxes_lp(input_video, interpolated_df, output_video_path):

    resolution = (1920,1080)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(interpolated_df)

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file: {input_video}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # You can change the codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, resolution)# (frame_width, frame_height))

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
            #print(_)
            lp_bbox_vals = row['lp_bbox'].split(' ')
            lp_bbox_vals[:]=map(lambda x: int(float(x)),lp_bbox_vals)
            lp_bbox_vals = str(lp_bbox_vals)
            lp_bbox_vals_str = "[" + lp_bbox_vals[1:-1] + "]"
            license_plate_bbox = ast.literal_eval(lp_bbox_vals_str)

            #H = license_plate_bbox[3] - license_plate_bbox[1]
            #W = license_plate_bbox[2] - license_plate_bbox[0]

            # Draw bounding boxes on the frame
            cv2.rectangle(frame, (int(license_plate_bbox[0]), int(license_plate_bbox[1])),
                          (int(license_plate_bbox[2]), int(license_plate_bbox[3])), (0, 0, 255), 2)

            # Show license plate text above the license plate bounding box
            if str(row['lp']) != ' ' and str(row['lp']) != '0':
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
            frame = cv2.resize(frame,resolution,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(frame)

        frame_number += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()


def WriteLog(df: pd.DataFrame, path: str, creation_datetime: datetime,fps: float):
    spf = 1/fps
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


def compress_video(video_full_path, output_file_name, target_size):
    # Reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
    min_audio_bitrate = 32000
    max_audio_bitrate = 256000

    probe = ffmpeg.probe(video_full_path)
    # Video duration, in s.
    duration = float(probe['format']['duration'])
    # Audio bitrate, in bps.
    audio_bitrate = float(next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)['bit_rate'])
    # Target total bitrate, in bps.
    target_total_bitrate = (target_size * 1024 * 8) / (1.073741824 * duration)

    # Target audio bitrate, in bps
    if 10 * audio_bitrate > target_total_bitrate:
        audio_bitrate = target_total_bitrate / 10
        if audio_bitrate < min_audio_bitrate < target_total_bitrate:
            audio_bitrate = min_audio_bitrate
        elif audio_bitrate > max_audio_bitrate:
            audio_bitrate = max_audio_bitrate
    # Target video bitrate, in bps.
    video_bitrate = target_total_bitrate - audio_bitrate

    i = ffmpeg.input(video_full_path)
    ffmpeg.output(i, os.devnull,
                  **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 1, 'f': 'mp4'}
                  ).overwrite_output().run()
    ffmpeg.output(i, output_file_name,
                  **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate}
                  ).overwrite_output().run()
    

def draw_bounding_boxes(input_video_path, csv_file_path, output_video_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
    
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
        ret=True
        while ret:
            ret, frame = cap.read()
    
            # Filter the DataFrame for the current frame number
            frame_data = df[df['fr_number'] == frame_number]        
    
            #print(frame_data)
            for _, row in frame_data.iterrows():

                #license plate
                lp_bbox_vals = row['lp_bbox'].split(' ')
                lp_bbox_vals[:]=map(lambda x: int(float(x)),lp_bbox_vals)
                lp_bbox_vals = str(lp_bbox_vals)
                lp_bbox_vals_str = "[" + lp_bbox_vals[1:-1] + "]"
                license_plate_bbox = ast.literal_eval(lp_bbox_vals_str)
    
                # Draw bounding boxes on the frame
                #cv2.rectangle(frame, (int(car_bbox[0]), int(car_bbox[1]), (int(car_bbox[2]), int(car_bbox[3]))), (0, 255, 0), 2)
                cv2.rectangle(frame, (int(license_plate_bbox[0]), int(license_plate_bbox[1])),
                              (int(license_plate_bbox[2]), int(license_plate_bbox[3])), (0, 0, 255), 1)
    
                lp_bbox_vals = row['lp_bbox'].split(' ')
                lp_bbox_vals[:]=map(lambda x: int(float(x)),lp_bbox_vals)
                #print(lp_bbox_vals)
                #x1, y1, x2, y2
                #H, W, _ = lp_bbox_vals.shape
                H = lp_bbox_vals[3]-lp_bbox_vals[1]
                W = lp_bbox_vals[2]-lp_bbox_vals[0]
                #print(H,W)
    
                try:
                    #frame[int(car_bbox_vals[1]) - H - 100:int(car_bbox_vals[1]) - 100,
                    #      int((car_bbox_vals[2] + car_bbox_vals[0] - W) / 2):int((car_bbox_vals[2] + car_bbox_vals[0] + W) / 2), :] = license_crop
    
                    #frame[int(car_bbox_vals[1]) - H - 400:int(car_bbox_vals[1]) - H - 100,
                    #      int((car_bbox_vals[2] + car_bbox_vals[0] - W) / 2):int((car_bbox_vals[2] + car_bbox_vals[0] + W) / 2), :] = (255, 255, 255)
    
                    license_crop = frame[lp_bbox_vals[1]+2:lp_bbox_vals[3],lp_bbox_vals[0]+2:lp_bbox_vals[2], :]
                    #frame[int(car_bbox_vals[1]) - H - 100:int(car_bbox_vals[1]) - 100,
                    #      int((car_bbox_vals[2] + car_bbox_vals[0] - W) / 2):int((car_bbox_vals[2] + car_bbox_vals[0] + W) / 2), :] = license_crop
                    #cv2.imshow('frame', license_crop)
                    #cv2.waitKey(0)
    
                    padding = 10
                    border_thickness = 1
                    (text_width, text_height), _ = cv2.getTextSize(
                        row['lp'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        5)
    
                    #background_width = car_bbox[2] - car_bbox[0]
                    #cv2.rectangle(frame, (car_bbox[0], car_bbox[1]-text_height-H), (car_bbox[2], car_bbox[1]), (0, 0, 0), thickness=cv2.FILLED)
                    #cv2.rectangle(frame, (text_x, text_y - text_height), (text_x + background_width, text_y), (255, 255, 255), thickness=cv2.FILLED)
    
                    #cv2.rectangle(frame, (text_x-padding, text_y-text_height-padding), (text_x + text_width+padding, text_y+padding), (0, 0, 0), thickness=cv2.FILLED)
                    #cv2.rectangle(frame, (car_bbox[0], car_bbox[1]-text_height-H), (car_bbox[2], car_bbox[1]), (0, 255, 0), thickness=border_thickness)
    
                    #frame[car_bbox_vals[1]+2:car_bbox_vals[3], car_bbox_vals[0]+2:car_bbox_vals[2], :] = license_crop
                    #frame[car_bbox[1]:car_bbox[1]+(lp_bbox_vals[3]-lp_bbox_vals[1]), car_bbox[0]:car_bbox[0]+(lp_bbox_vals[2]-lp_bbox_vals[0]), :] = license_crop
                    #x=100
                    #y=100
                    #alpha = 1.0  # Controls the transparency of the cropped image
                    #beta = 1.0 - alpha
                    #cropped_height = lp_bbox_vals[3] - lp_bbox_vals[1] - 2
                    #cropped_width = lp_bbox_vals[2] - lp_bbox_vals[0] - 2
    
                    #cv2.addWeighted(frame, alpha, license_crop, beta, 0, frame[y:y + cropped_height, x:x + cropped_width])
    
                    if(row['lp'] != '0'):
                        cv2.putText(frame,
                                row['lp'],
                                (int((license_plate_bbox[0] + license_plate_bbox[2] - text_width) / 2), int(license_plate_bbox[1] - H + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (255, 255, 255),
                                3)
    
                except:
                    pass
    
            # Write the frame with bounding boxes to the output video
            #try:
            if frame is not None:
                frame = cv2.resize(frame,(1920,1080),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                out.write(frame)
            #except:
            #    pass
            frame_number += 1
            #frame = cv2.resize(frame, (1280, 720))
    
            #cv2.imshow('frame', frame)
            #cv2.waitKey(0)
    
        # Release the video capture and writer objects
        cap.release()
        out.release()