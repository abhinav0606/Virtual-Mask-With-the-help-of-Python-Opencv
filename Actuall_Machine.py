import cv2
import numpy as np
faces=cv2.CascadeClassifier("frontal_face.xml")
capture=cv2.VideoCapture(0)
s_img = cv2.imread("output.png",-1)
# print(s_img.shape)
s_img=cv2.resize(s_img,(125,125))
# print(s_img.shape)
alpha_s = s_img[:, :, 3] / 255.0
alpha_l = 1.0
while True:
    ret,frame=capture.read()
    # print(frame.shape)
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=faces.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in face:
        print(frame[y+50:y+h,x:x+w].shape)
        s_img=cv2.resize(s_img,(frame[y+60:y+h+10,x:x+w].shape[1],frame[y+60:y+h+10,x:x+w].shape[0]))
        # frame[y:y+h,x:x+w]=s_img
        print(s_img.shape)
        print("Hello")
        alpha_s = s_img[:, :, 3] / 255.0
        alpha_l = 1.0-alpha_s
        for i in range(3):
            frame[y+60:y+h+10,x:x+w,i] = (alpha_s * s_img[:, :,i] +alpha_l * frame[y+60:y+h+10, x:x+w,i])
    cv2.imshow("Abhinav",frame)
    if cv2.waitKey(1)==13:
        break
capture.release()
cv2.destroyAllWindows()