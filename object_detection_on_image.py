import cv2
img = cv2.imread('images/image.png')

classnames = []  
classfile = 'files/things.names.txt'

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
    
p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(p, v)
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.4)

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    if 1 <= classId <= len(classnames):  
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
        cv2.putText(img, classnames[classId - 1],
                    (box[0] + 10, box[1] + 20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)
    else:
        print(f"Invalid classId: {classId}")

cv2.imshow('Rakwan', img)
cv2.waitKey(0)
