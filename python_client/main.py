# -*-coding:utf-8 -*-
# pip install opencv-python
import cv2
import time
from datetime import datetime
from threading import Thread

frame_current = ""


def rtsp_read():
    global frame_current
    rtsp_url = "rtsp://liuyang:liuyang@192.168.1.107:8554/live"
    capture = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = capture.read()
        frame_current = frame
        # cv2.imshow("frame", frame_current)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    capture.release()


def rtsp_frame_save():
    global frame_current
    while True:
        try:
            time_str = datetime.now().strftime("%Y%m%d%H%M%S")
            img_path = "./img/" + time_str + ".png"
            if type(frame_current) != str:
                is_writed = cv2.imwrite(img_path, frame_current)
                print(img_path + " writed ? " + str(is_writed))
                frame_current = ""
            else:
                print(img_path + " frame is empty")
        except:
            print("someexception")
        time.sleep(2)

if __name__ == '__main__':
    read_procrss = Thread(target=rtsp_read)
    save_process = Thread(target=rtsp_frame_save)

    print("start all pricess")
    read_procrss.start()
    save_process.start()
