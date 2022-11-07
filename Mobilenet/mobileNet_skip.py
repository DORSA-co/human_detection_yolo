import time
import cv2
import object_detection
from motion_detection import * 
import numpy as np
from skipFrame import VideoCam



VIDEO_SOURCE_WEBCAM = False

if __name__ == '__main__':
    # object detection module
    object_detection_obj = object_detection.Object_Detector(config_path='Mobilenet/cfgs/ssd_mobilenet_v3_large_coco.pbtxt',
                                                            model_path='Mobilenet/cfgs/ssd_mobilenet_v3_large_coco.pb',
                                                            classes_path='Mobilenet/cfgs/coco.names')
    SKIPFRAME = 5
    url = '../videos/XVR_CH_1.mp4'
    v1 = VideoCam(url)
    v1.check_camera(v1.cap)
    ret = v1.cap.grab()
    ret, frame = v1.get_frame()
    # motion_detector_obj = Motion_detection()
    # flag, frame, bboxes, color, confidence, string_classes_found = object_detection_obj.detect(frame=frame.copy())
    # counter = 0
    # roi_selected = cv2.selectROI(frame, False)
    # print(roi_selected)
    
    ct = 0
    while True:
        ct += 1
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
                # detection
                start = time.time()
                # y = 500
                # y2 = 1500
                # x = 0
                # x2 = 1500
                # res, frame = motion_detector_obj.detect_motion_with_show(frame=frame)
                # if res:
                # print(res)
                # flag, frame, bboxes, color, confidence, string_classes_found = object_detection_obj.detect(frame=frame.copy())
                roi = frame#[roi_selected[1]:roi_selected[3], roi_selected[0]:roi_selected[2]]
                
                flag, frame, bboxes = object_detection_obj.detect(frame=roi.copy())
                # counter = 0
                # for box in bboxes:
                #     x, y, w, h = box[0], box[1], box[2], box[3]
                #     cv2.rectangle(frame, (x,y), (x+w,y+h), color[counter], thickness=2)
                #     display_text = '{}:{:.2f}'.format(string_classes_found[counter], confidence[counter])
                #     cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 3, color[counter], 2)
                #     counter = counter + 1
                fps =round((1/(time.time() - start)), 2)
                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fps)), (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
                v1.show_frame(frame_resized, 'frame')
                
        except KeyboardInterrupt:
            v1.close_cam()
            exit(0)