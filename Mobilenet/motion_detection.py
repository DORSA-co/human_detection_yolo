import cv2
import numpy as np
import time



class Motion_detection():
    def __init__(self, diff_threshold=50, area_threshold=50):
        self.prev_frame = None
        self.diff_threshold = diff_threshold
        self.area_threshold = area_threshold


    def detect_motion(self, frame):
        # prepare frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(src=frame, ksize=(5,5), sigmaX=0)

        # First frame; there is no previous one yet
        if self.prev_frame is None:
            self.prev_frame = frame
            return False
        
        else:
            # calculate difference and update previous frame
            diff = cv2.absdiff(src1=self.prev_frame, src2=frame)
            self.prev_frame = frame

            # Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff = cv2.dilate(diff, kernel, 1)

            # Only take different areas that are different enough (>20 / 255)
            diff = cv2.threshold(src=diff, thresh=self.diff_threshold, maxval=255, type=cv2.THRESH_BINARY)[1]

            # find contours of motion detected arears
            contours, _ = cv2.findContours(image=diff, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # no motion detected
            if len(contours) == 0:
                return False
            else:
                return True
    

    def detect_motion_with_show(self, frame):
        # prepare frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(src=gray_frame, ksize=(5,5), sigmaX=0)

        # First frame; there is no previous one yet
        if self.prev_frame is None:
            self.prev_frame = gray_frame
            return False, frame
        
        else:
            # calculate difference and update previous frame
            diff = cv2.absdiff(src1=self.prev_frame, src2=gray_frame)
            self.prev_frame = gray_frame

            # Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff = cv2.dilate(diff, kernel, 1)

            # Only take different areas that are different enough (>20 / 255)
            diff = cv2.threshold(src=diff, thresh=self.diff_threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
            
            # remove topright clock on image
            diff[:100] = 0

            # find contours of motion detected arears
            contours, _ = cv2.findContours(image=diff, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # draw motion detected areas on image
            for contour in contours:
                # too small: skip!
                if cv2.contourArea(contour) < self.area_threshold:
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=1)

            # no motion detected
            if len(contours) == 0:
                return False, frame
            else:
                return True, frame



if __name__ == '__main__':
    # motion detection object
    motio_detector_obj = Motion_detection()

    # video
    cap = cv2.VideoCapture('../videos/XVR_CH_7.mp4')

    while True:
        success, frame = cap.read()
        if not success:
            break

        start = time.time()
        res, out_frame = motio_detector_obj.detect_motion_with_show(frame=frame)
        print(res)
        print('FPS: %s' % (1/(time.time() - start)))
        frame_resized = cv2.resize(frame, None, fx = 0.5, fy=0.5)
        cv2.imshow('Motion detection', frame_resized)
        cv2.waitKey(30)


