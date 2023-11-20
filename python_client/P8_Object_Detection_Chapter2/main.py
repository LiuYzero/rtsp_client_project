# -*-coding:utf-8-*-
import cv2
import numpy as np

# img = cv2.imread('test_person.png')

thres = 0.5
nms_threshold = 0.5

# rtsp_url = "rtsp://liuyang:liuyang@192.168.1.107:8554/live"
# cap = cv2.VideoCapture(rtsp_url)
cap = cv2.VideoCapture('test_vlog1.mp4')
cap.set(3,640)
cap.set(4,480)
cap.set(10,150)



class_names = []
class_file = 'coco.names'
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'
net = cv2.dnn.DetectionModel(weights_path,config_path)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    is_success, img = cap.read()
    class_ids, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float, confs))
    # print(type(confs[0]))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold=nms_threshold)
    print(indices)
    print("class_ids: ",class_ids)


    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,y+h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, class_names[class_ids[i]-1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(img, str(confs[i])[:4], (box[0] + 10, box[1] + 60), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 255, 0), 2)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()