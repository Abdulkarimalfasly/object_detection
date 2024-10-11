import cv2
import numpy as np

classnames = []  
classfile = 'files/things.names.txt'
with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(p, v)
net.setInputSize(320, 320)  
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to read from camera")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=0.3)

    if len(classIds) > 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            
            if 1 <= classId <= len(classnames):  
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                cv2.putText(img, f"{classnames[classId - 1]}: {confidence:.2f}",
                            (box[0] + 10, box[1] + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)

    cv2.imshow('Object Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
