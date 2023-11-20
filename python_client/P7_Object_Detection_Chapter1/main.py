# -*-coding:utf-8-*-
import cv2

# img = cv2.imread('test_person.png')

# rtsp_url = "rtsp://liuyang:liuyang@192.168.1.107:8554/live"
# cap = cv2.VideoCapture(rtsp_url)
cap = cv2.VideoCapture('test_animals.mp4')
cap.set(3,640)
cap.set(4,480)

class_names = []
class_file = 'coco.names'
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
print (class_names)

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'
net = cv2.dnn.DetectionModel(weights_path,config_path)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    is_success, img = cap.read()
    class_ids, confs, bbox = net.detect(img, confThreshold=0.5)
    print(class_ids)

    if len(class_ids) != 0:
        for class_id, confidence, box in zip(class_ids, confs, bbox):
            # print (class_id, confidence, box)
            cv2.rectangle(img,box, color=(0,255,0),thickness=2)
            cv2.putText(img, class_names[class_id-1].upper(), (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 2)
            cv2.putText(img, str(confidence)[:4], (box[0]+10, box[1] +60), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
cap.release()