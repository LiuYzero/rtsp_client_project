# -*-coding:utf-8 -*-
import cv2
import numpy as np
import face_recognition

# load picture
imgElon = face_recognition.load_image_file("ImagesBasic/ElonMusk.png")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgElonTest = face_recognition.load_image_file("ImagesBasic/ElonMuskTest.png")
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

imgBillGates = face_recognition.load_image_file("ImagesBasic/BillGates.png")
imgBillGates = cv2.cvtColor(imgBillGates, cv2.COLOR_BGR2RGB)

# detection face
face_location = face_recognition.face_locations(imgElon)[0]
encode_elon = face_recognition.face_encodings(imgElon)[0]

face_test_location = face_recognition.face_locations(imgElonTest)[0]
encode_elon_test = face_recognition.face_encodings(imgElonTest)[0]

face_bill_location = face_recognition.face_locations(imgBillGates)[0]
encode_bill = face_recognition.face_encodings(imgBillGates)[0]

# draw rectangle
print (face_location)
print (encode_elon)
cv2.rectangle(imgElon,(face_location[3],face_location[0]),(face_location[1],face_location[2]),
                (255,0,255), 2)
cv2.rectangle(imgElonTest,(face_test_location[3],face_test_location[0]),(face_test_location[1],face_test_location[2]),
                (255,0,255), 2)
cv2.rectangle(imgBillGates,(face_bill_location[3],face_bill_location[0]),(face_bill_location[1],face_bill_location[2]),
                (255,0,255), 2)

# calc distance between elon and elon_test ;-) it's fun
known_encodes = [encode_elon, encode_bill]
distance_result = face_recognition.compare_faces(known_encodes, encode_elon_test)
distance_value = face_recognition.face_distance(known_encodes, encode_elon_test)
print (distance_result)
print (distance_value)

# draw title
cv2.putText(imgElonTest, ""+str(distance_result[0])+" "+str(round(distance_value[0],2)),
        (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
cv2.putText(imgBillGates, ""+str(distance_result[1])+" "+str(round(distance_value[1],2)),
        (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

# show image
cv2.imshow("ElonMusk", imgElon)
cv2.imshow("ElonMusk Test", imgElonTest)
cv2.imshow("Bill Gates", imgBillGates)

cv2.waitKey(0)