from infer_yolo import Model
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time

model = Model()
cap = cv2.VideoCapture(0)
start = time.time()
while True:
    _, frame = cap.read()
    model.infer(frame)
    print(time.time()- start)
    start = time.time()

