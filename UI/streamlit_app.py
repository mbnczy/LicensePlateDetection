# %%
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, '/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/Logic')
import main_logic as logic


#   cmd line tool for desktop app from streamlit website: 
#       nativefier --name "LicensePlateDetector" "http://localhost:8515" --platform "mac"



def main():
    st.set_page_config(layout="wide")

    st.title("License Plate Prediction App")

    #st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        newpath = os.path.join("/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/video_data",f'{uploaded_file.name[0:-4]}_uploaded.mp4')
        with open(newpath,"wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Video uploaded", icon='✅')
    params_col1, params_col2 = st.columns(2)
    with params_col1:    
        st.header("Choose Model")
        model_name = st.selectbox("Select a model", 
                                  ["Yolo v8 - M (120)",
                                    "Yolo v8 - N (120)",
                                    "Yolo v8 - N (90)",
                                    "Yolo v8 - N (60)",
                                    "Yolo v8 - N (30)"])
    with params_col2:
        st.header('Params')
        im_type = st.selectbox("Image manipulation strength", ["weak","mid","strong"])
        bestscore = st.checkbox("Show only the best scored lp for each frame")
        
        
    btn_predict = st.button("Predict")
    
    if btn_predict:
        if uploaded_file is not None:
            #video_bytes = uploaded_file.read()
            #video_np = np.frombuffer(video_bytes, np.uint8)
            #input_video = cv2.imdecode(video_np, cv2.IMREAD_COLOR)
            output_video_path = perform_prediction(newpath, model_name,bestscore,im_type)
            
            #st.video(output_video.tobytes())testvideo_path[0:-4]+'_lp.avi'
            st.video(output_video_path)
            st.success('Prediction complete!', icon='✅')
            st.header('log')
            st.dataframe(pd.read_csv('/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/time_log.csv'))
            os.remove('/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/time_log.csv')
            os.remove('/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/logs/log_interpolated.csv')
            #os.remove(output_video_path)
            os.remove(os.path.join("/Users/banoczymartin/Library/Mobile Documents/com~apple~CloudDocs/OE/platedetector/video_data",f'{uploaded_file.name[0:-4]}_uploaded.mp4'))
def perform_prediction(localpath, model_name, showonlybestconf,im_type):
    with st.spinner('Predicting...'):
        logic.Run(localpath, model_name, showonlybestconf, im_type)


if __name__ == "__main__":
    main()



# %%

# %%
