
import numpy as np
import cv2
import time


CLASSES_TO_DETECT = [1, 3]
np.random.seed(9)


class Object_Detector():
    def __init__(self, config_path, model_path, classes_path, input_size=416, confidence_threshold=0.5):
        self.config_path = config_path
        self.model_path = model_path
        self.classes_path = classes_path
        self.input_size = input_size
        self.conf_threshold = confidence_threshold
        self.nms_threshold = 0.2 # (0.1 to 1) 1 means no suppress , 0.1 means high suppress 
        self.target_classes = []
        #
        self.build_model()
        self.load_classes()
        self.choose_target_classes()
    

    def build_model(self):
        self.net = cv2.dnn_DetectionModel(self.model_path, self.config_path)
        self.net.setInputSize(self.input_size, self.input_size)
        self.net.setInputScale(1.0/ 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    
    def load_classes(self):
        with open(self.classes_path, 'r') as f:
            self.classNames = f.read().splitlines()
        f.close()
        # colors
        self.colors = np.random.uniform(0, 255, size=(len(self.classNames), 3))

    
    def choose_target_classes(self, target_classes=CLASSES_TO_DETECT):
        self.target_classes = target_classes

    
    def change_classids(self, class_ids):
        for i in range(len(class_ids)):
            # boat, truck, train to car
            if class_ids[i] in [9, 8, 7, 6]:
                class_ids[i] = 3

        return class_ids

    
    def detect(self, frame):
        # detection
        # print('call detection')
        class_ids, confs, bboxes = self.net.detect(frame, confThreshold=self.conf_threshold)
        # print('here')
        
        # return if no object detected
        if len(class_ids) == 0:
            color = (0, 0, 0)
            confidence = 0
            string_classes_found = []
            return False, frame, bboxes#, color, confidence, string_classes_found

        # change some classes
        class_ids = self.change_classids(class_ids=class_ids)

        bboxes = list(bboxes)
        
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float, confs))
        # print('here2')
        indices = cv2.dnn.NMSBoxes(bboxes, confs, self.conf_threshold, self.nms_threshold)

        flag = False
        string_classes_found = []
        colors = []
        confidences = []
        boxes = []
        # print('here3')
        if len(class_ids) != 0:
            for i in indices:
                if class_ids[i] not in self.target_classes:
                    continue
                
                flag = True
                box = bboxes[i]
                boxes.append(box)
                confidence = round(confs[i],2)
                color = self.colors[class_ids[i]-1]
                confidences.append(confidence)
                colors.append(color)
                string_classes_found.append(self.classNames[class_ids[i]-1])
                x, y, w, h = box[0], box[1], box[2], box[3]
                #
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, thickness=2)
                display_text = '{}:{:.2f}'.format(self.classNames[class_ids[i]-1], confidence)
                cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
  
        return flag, frame, boxes#, colors, confidences, string_classes_found



if __name__ == '__main__':
    # object detection module
    print('detection test')
    object_detection_obj = Object_Detector(config_path='cfgs/ssd_mobilenet_v3_large_coco.pbtxt',
                                            model_path='cfgs/ssd_mobilenet_v3_large_coco.pb',
                                            classes_path='cfgs/coco.names')

    # video
    cap = cv2.VideoCapture('../videos/XVR_CH_1.mp4')

    while True:
        success, frame = cap.read()
        if not success:
            break

        start = time.time()
        flag, frame, bboxes = object_detection_obj.detect(frame=frame)
        # print(bboxes)
        print('FPS: %s' % (1/(time.time() - start)))

        # cv2.imshow('Object detection', out_frame)
        cv2.waitKey(20)


