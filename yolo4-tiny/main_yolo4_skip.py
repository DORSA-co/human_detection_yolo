import cv2
from gui_buttons import Buttons
import time
from skipFrame import VideoCam

# # Initialize Buttons
button = Buttons()
colors = button.colors


# Opencv DNN
net = cv2.dnn.readNet("../Object-Detection/object_detection_crash_course/dnn_model/yolov4-tiny.weights", 
"../Object-Detection/object_detection_crash_course/dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("../Object-Detection/object_detection_crash_course/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects list")
print(classes)

SKIPFRAME = 5
url = '../videos/XVR_CH_All.mp4'
v1 = VideoCam(url)
v1.check_camera(v1.cap)

ct = 0
while True:
    ct += 1
    try:
        ret = v1.cap.grab()
        if ct % SKIPFRAME == 0:  # skip some frames
            ret, frame = v1.get_frame()
            if not ret:
                v1.restart_capture(v1.cap)
                v1.check_camera(v1.cap)
                continue
            # frame HERE

            start = time.time()
            # Object Detection
            (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.5, nmsThreshold=.4)
            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                (x, y, w, h) = bbox
                class_name = classes[class_id]
                color = colors[class_id]
                # if class_name in active_buttons:
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            fps = round((1/(time.time() - start)), 2)
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            print('fps: ', fps)
            frame_resized = cv2.resize(frame, None, fx = 0.4, fy=0.4)
            v1.show_frame(frame_resized, 'frame')
    except KeyboardInterrupt:
        v1.close_cam()
        exit(0)
# # Initialize camera
# cap = cv2.VideoCapture(0)#('../videos/XVR_CH_1.mp4') #webcam 0
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# # FULL HD 1920 x 1080
# # Create window
# cv2.namedWindow("Frame")

# while True:
# # Get frames
#     ret, frame = cap.read()
#     start = time.time()
#     # Object Detection
#     (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.5, nmsThreshold=.4)
#     for class_id, score, bbox in zip(class_ids, scores, bboxes):
#         (x, y, w, h) = bbox
#         class_name = classes[class_id]
#         color = colors[class_id]

#         # if class_name in active_buttons:
#         cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
#         fps =round((1/(time.time() - start)), 2)
#         # Display FPS on frame
#         cv2.putText(frame, "FPS : " + str(int(fps)), (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
#         print('fps: ', fps)

#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break


