import time
import cv2
import object_detection_roi
from motion_detection import * 
import numpy as np
from skipFrame import VideoCam
from deep_sort_realtime.deepsort_tracker import DeepSort



VIDEO_SOURCE_WEBCAM = False

if __name__ == '__main__':
    # object detection module
    object_detection_obj = object_detection_roi.Object_Detector(config_path='Mobilenet/cfgs/ssd_mobilenet_v3_large_coco.pbtxt',
                                                            model_path='Mobilenet/cfgs/ssd_mobilenet_v3_large_coco.pb',
                                                            classes_path='Mobilenet/cfgs/coco.names')
    
    SKIPFRAME = 5 ##### making it 10 frames per second
    
    video_files = ['XVR_CH_1.mp4', 'XVR_CH_2.mp4', 'XVR_CH_3.mp4', 
    'XVR_CH_4.mp4', 'XVR_CH_5.mp4', 'XVR_CH_6.mp4', 'XVR_CH_7.mp4', 
    'XVR_CH_8.mp4', 'XVR_CH_9.mp4']
    
    area1 = [(100, 220), (250, 1073), (1920, 1080), (1920, 435), (1200, 220)]
    area2 = [(0, 420), (240, 1080), (1920, 1080), (1920, 880), (320, 330)]
    area3 = [(0, 420), (240, 1080), (1920, 1080), (1920, 880), (320, 330)]
    area4 = [(0, 260), (0, 1080), (1180, 1080), (360, 260)]
    area5 = [(0, 260), (0, 1080), (1180, 1080), (360, 260)]
    area6 = [(260, 360), (75, 1080), (1920, 1080), (1920, 660), (1450, 470), (910, 560), (510, 260)]
    area7 = [(0, 900), (0, 1080), (1920, 1080), (1100, 520), (795, 400), (620, 420)]
    area8 = [(1070, 230), (1200, 230), (1920, 1080), (730, 1080)]
    area9 = [(430, 1080), (1250, 460), (1380, 460), (1820, 1080)]
    roi_of_each_video = [area1, area2, area3, area4, area5, area6, area7, area8, area9]

    for video in video_files:

        url = '../videos/' + video #### getting te video file
        roi_selected = roi_of_each_video[video_files.index(video)]  #### getting its roi

        v1 = VideoCam(url) #### open the file
        v1.check_camera(v1.cap)     #### check camera
        ret = v1.cap.grab()         
        ret, frame = v1.get_frame()
        ct = 0
        while True:
            ct += 1  ### count frames to skip
            try:
                ret = v1.cap.grab()
                ret, frame = v1.get_frame()
                if ct % SKIPFRAME == 0:  # skip some frames and only detect in some frames
                    ret, frame = v1.get_frame()
                    if not ret:
                        v1.restart_capture(v1.cap)
                        v1.check_camera(v1.cap)
                        continue
                    # frame HERE
                    ##### DeepSORT Tracking
                    # tracker = DeepSort(max_age=5)
                    # bbs = object_detector.detect(frame) # your own object detection
                    start = time.time()
                    flag, frame, bboxes = object_detection_obj.detect(roi_selected, frame=frame.copy()) 
                    # tracks = tracker.update_tracks(bboxes, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class ), also, no need to give frame as your chips has already been embedded
                    # for track in tracks:
                    #     print('track:', track)
                    #     if not track.is_confirmed():
                    #         continue
                    #     track_id = track.track_id
                    #     print('track_id', track_id)
                    #     # cv2.putText(frame, track_id, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
                    #     ltrb = track.to_ltrb()
                    #     print('ltrb', ltrb)
                    # detection
                    
                    cv2.polylines(frame, [np.array(roi_selected, np.int32)], True, (255, 0, 0), 6)                   
                    
                    fps =round((1/(time.time() - start)), 2)
                    # Display FPS on frame
                    cv2.putText(frame, "FPS : " + str(int(fps)), (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
                    v1.show_frame(frame_resized, 'frame')
                    
            except KeyboardInterrupt:
                v1.close_cam()
                exit(0)