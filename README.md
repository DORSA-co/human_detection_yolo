this repository is designated for the DVR project in Dorsa Co. there have been some different models tested including:

MobileNet ssd_mobilenet_v3_large_coco, yolov3-tiny, yolov4-tiny, yolov5

the following pros and cons of each model will be explained:

1.	MobileNet: has acceptable speed but detects some wrong objects in the background -> solution: selecting an ROI might help, speed is improved using skip frame
2.	yolov3-tiny: too slow and too many wrong detections
3.	yolov4-tiny: speed is neither good nor bad, detection is mostly fine (but not always!!)
4.	yolov4-tiny with tracking using deep SORT algorithm: SORT(Simple Online Realtime Tracking) is a tracking algorithm, it makes the whole thing very slow but can be helpful and is the most accurate and fast-tracking at the time. DeepSORT:https://arxiv.org/abs/1703.07402
5.	yolov5: https://github.com/ultralytics/yolov5 -> has 5 different weight models: n, s, m, l, and x (from smallest to largest) this one has acceptable speed and accuracy

testing the models on RaspberryPi2:

we have tested these on RaspberryPi2 2Gig, speed drops in comparison to running on a corei7 laptop

to run these on a RaspberianOS, OpenCV and python<=3.8 and PyTorch must be installed:

PyTorch: it can't be installed with pip and the normal wheel of python-pip since the CPU is arm-based and not intel

I installed the wheel from: https://github.com/ljk53/pytorch-rpi

Pytorch version must be higher than v1.7 for yolov5

this was also tested on yolov4-tiny CPP, which had a way worst speed

ATTENTION: this project does not have a UI yet, but it will soon.

