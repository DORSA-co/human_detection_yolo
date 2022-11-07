import time
import cv2
from imutils.video import VideoStream
import motion_detection, object_detection
from EuclideanDistTracker_class import *
import numpy as np

VIDEO_SOURCE_WEBCAM = False
# Create tracker object
tracker = EuclideanDistTracker()
if not VIDEO_SOURCE_WEBCAM:
    cap = cv2.VideoCapture('../videos/XVR_CH_8.mp4')
else:
    cap = VideoStream(src=0).start()


if __name__ == '__main__':
    # object detection module
    object_detection_obj = object_detection.Object_Detector(config_path='Mobilenet/cfgs/ssd_mobilenet_v3_large_coco.pbtxt',
                                                            model_path='Mobilenet/cfgs/ssd_mobilenet_v3_large_coco.pb',
                                                            classes_path='Mobilenet/cfgs/coco.names')
    
    # motion detection object
    # motion_detector_obj = motion_detection.Motion_detection()
    while True:
        if not VIDEO_SOURCE_WEBCAM:
            success, frame = cap.read()
            
            if not success:
                break
        else:
            frame = cap.read()
        
        start = time.time()

        # check if any motion detected
        # res_m = motion_detector_obj.detect_motion(frame=frame.copy())
        detections = []
        # if res_m:
        # detection
        res_d, detected_frame, bboxes = object_detection_obj.detect(frame=frame.copy())
        if bboxes:
            for i in range(len(bboxes)):
                x, y, w, h = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                detections.append([x, y, w, h])
            
        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(detected_frame, str(id), (x- 15, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if res_d:
            fps =round((1/(time.time() - start)), 2)
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            print('fps: ', fps)
            frame_resized = cv2.resize(detected_frame, None, fx = 0.4, fy=0.4)
            cv2.imshow('Object detection', frame_resized)
        
        else:
            cv2.putText(frame, "Tracking failure detected", (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            frame_resized = cv2.resize(frame, None, fx = 0.4, fy=0.4)
            cv2.imshow('Object detection', frame_resized)
            for i in range(10):
                frame = cap.read()

        
        cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()    
