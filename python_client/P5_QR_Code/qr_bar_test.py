# -*-coding:utf-8-*-

import cv2
import numpy as np
from pyzbar.pyzbar import decode

rtsp_url = "rtsp://liuyang:liuyang@192.168.1.107:8554/live"
cap = cv2.VideoCapture(rtsp_url)
cap.set(3,640)
cap.set(4,480)

while True:
    is_success, img = cap.read()
    for barcode in decode(img):
        qr_data = barcode.data.decode('utf-8')
        print (qr_data)
        points = np.array([barcode.polygon], np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(img, [points],True, (255,0,255), 5)
        points2 = barcode.rect
        cv2.putText(img, qr_data, (points2[0],points2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(255,0,255), 2)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()



