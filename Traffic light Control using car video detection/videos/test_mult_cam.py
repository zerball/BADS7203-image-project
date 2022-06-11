import threading
import time
import serial
from serial import Serial


import math
import numpy as np
import cv2


# def printit():
  # threading.Timer(n, printit).start()
  # print("Hello, World!" ,time.time())

Video_list = [1,2,3,4]
delay_time = [3,3,3,3]
j=0

byte = 41

send_list = ['A','B','C','D']
#out = serial.Serial("/dev/ttyS0")  # "COM1" on Windows
#out.write(bytes(byte))
#ser = serial.Serial('COM3', 9600)
i=0


cam1 = cv2.VideoCapture("./T1_2.mp4")
cam2 = cv2.VideoCapture("./T2_2.mp4")
cam3 = cv2.VideoCapture("./T3_1.mp4")
cam4 = cv2.VideoCapture("./T4_2.mp4")




while True:
    
    ret1, cam11 = cam1.read()
    ret2, cam21 = cam2.read()
    
    ret3, cam31 = cam3.read()
    ret4, cam41 = cam4.read()
    
    
    
    if (ret1):
        # Resize to make sure your com. will stay alive
        width, height, layers = cam11.shape
        new_w = width // 2
        new_h = height // 2
        re1 = cv2.resize(cam11,(new_h,new_w))
        
    
    if (ret2):
        # Resize to make sure your com. will stay alive
        width, height, layers = cam21.shape
        new_w = width // 2
        new_h = height // 2
        re2 = cv2.resize(cam21,(new_h,new_w))
        
        
    if (ret3):
        # Resize to make sure your com. will stay alive
        width, height, layers = cam31.shape
        new_w = width // 2
        new_h = height // 2
        re3 = cv2.resize(cam31,(new_h,new_w))
        
        
    if (ret4):
        # Resize to make sure your com. will stay alive
        width, height, layers = cam41.shape
        new_w = width // 2
        new_h = height // 2
        re4 = cv2.resize(cam41,(new_h,new_w))
        
    
    
        
        
        
    cam_list = [re1,re2,re3,re4]
    
    
     
        # #ser.write(bytes(byte))
    # ser.write(str.encode('C'))
    # ser.write(str.encode(send_list[i]))
    
    # time.sleep(1)
    # print(i)    
    # i+=1
    # if i%4 == 0:
        # i=0

    


# for i in Video_list:
    # print(i)
    # print(time.time()%60)
    # time.sleep(delay_time[j])
    # j+=1

    # cv2.imshow( "Color segmentation", np.hstack((cam11,cam21)))
    # cv2.imshow( "Color segmentation", np.hstack((cam31,cam41)))
    
    hor1 = np.hstack((re1,re2))
    hor2 = np.hstack((re3,re4))
    
    result = np.vstack((hor1,hor2))

    cv2.imshow("Result",result)
    
    if cv2.waitKey(100) == 27: # ESC to exit
        break
    
    for vd in cam_list:
        cv2.imshow("Working cam",vd)
        if cv2.waitKey(100) == 27: # ESC to exit
            break
            
        time.sleep(delay_time[j])
        j+=1
        if j%3 == 0:
            j=0
        
    if cv2.waitKey(100) == 27: # ESC to exit
        break
    
    
    
    


