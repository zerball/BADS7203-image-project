
# RUN THE CODE:
# On cmd line
#  >>> python trffic_light_control_using_video_analytics.py -y yolo --output output --skip-frames 10 <<<
# importing all modules (pip install -r "requirements.txt")

from asyncio.windows_events import NULL
from re import I
from telnetlib import theNULL
from imutils.video import FPS, fps
from pandas import isnull
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
import math as maths
import numpy as np

from cv2 import namedWindow
import glob

import serial
from serial import Serial

# command line argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, type=str,
                help="path to yolo directory")
# ap.add_argument("-i", "--input", required=True, type=str,
#                 help="path to input video file")
ap.add_argument("-o", "--output", required=True, type=str,
                help="path to output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.90,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
                help="number of frames to skip between detections"
                     "the higher the number the faster the program works")
args = vars(ap.parse_args())


# Function draws car centroid
# Centroid is green if the car is moving
# Centroid is red if the car is not moving
def draw_centroids(frame, objects, trackableObjects, long_stopped_cars, ClassStr):
    ClassStr = ClassStr
    objects_list = {}
    for (objectID, centroid) in objects.items():
        # check if a trackable objects exists for particular ID
        to = trackableObjects.get(objectID, None)

        # if it doesn't then we create a new one corresponding to the given centroid
        if to is None:
            to = TrackableObject(objectID, centroid)

        # place the trackable object into the dict.
        trackableObjects[objectID] = to

        # drawing circle and text
        if objectID in long_stopped_cars:
            # text = "ID {} STOPPED".format(objectID + 1)
            text = "ID {} STANDING".format(objectID + 1)
            # if a car is not moving then we draw a large yellow centroid
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 6, (0, 255, 255), -1)
        else:
            text = "ID {}".format(objectID + 1)
            # else we draw a smaller green centroid
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 3, (0, 255, 0), -1)
        
        ## Bally modify ##
        objects_list[objectID+1] = []
        objects_list[objectID+1].append(centroid.tolist()) ## append value centroids x,y
        objects_list[objectID+1].append(ClassStr) ## append class car

    return objects_list


# Function calculates distance between two centroids
def find_distance(c1, c2):
    c1 = c1.tolist()
    c2 = c2.tolist()
    return int(maths.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2))

# Function compares coords on the certain car's box on N and N+1 frames
# It returns a list of cars that are, PERHAPS, not moving
def compare_trackers(old_trackers, new_trackers, frame_width, stopped_car_IDs):
    for (old_objectID, old_centroid) in old_trackers.items():
        for (new_objectID, new_centroid) in new_trackers.items():
            if old_objectID == new_objectID:
                distance = find_distance(old_centroid, new_centroid)
                # print(f"Distance between centroids of car number {old_objectID+1} is {distance}")
                # If the distance between centroids is less than 1/N of width of the frame then we add it to the list
                if distance < frame_width / parts:
                    # print("it is OK")
                    if new_objectID not in stopped_car_IDs:
                        # print(f"{new_objectID + 1} is a new car - add it to stopped_car_ID")
                        stopped_car_IDs.append(new_objectID)
                else:
                    # print("it is more than we need")
                    if new_objectID in stopped_car_IDs:
                        # print(f"deleting {new_objectID + 1}")
                        # If the distance is more than 1/N then it means that the car started moving again - delete it from the list
                        stopped_car_IDs.remove(new_objectID)
            # if a car has moved away from the frame and we can not see it anymore then we should
            # delete it from the list of stopped cars
            if old_objectID not in new_trackers.keys():
                if old_objectID in stopped_car_IDs:
                    # print(f"car {old_objectID+1} is not on the frame anymore - deleting it...")
                    stopped_car_IDs.remove(old_objectID)
    # if new_trackers are an empty array, that means that there are NO cars of a frame at all
    # so we should clear stopped_car_IDs
    if len(new_trackers.keys()) == 0:
        if stopped_car_IDs != []:
            # print("there is no car on a frame - clear stopped_car_IDs")
            stopped_car_IDs.clear()

    return stopped_car_IDs


# Function finds cars that were not moving long enough
def find_stopped_cars(counting_frames, frames_to_stop):
    long_stopped_cars = []
    for ID, frames in counting_frames.items():
        if frames > frames_to_stop: # this number can be changed to increase work efficiency
            long_stopped_cars.append(ID)
    return long_stopped_cars

########################################################################################################################

# delay_time = [3,3,3,3]
# j = 0
# ser = serial.Serial('COM3', 9600)
Total_run_Frame = 100
FixLight = {"videos/cam_A.mp4" : 'A',"videos/cam_B.mp4": 'B',"videos/cam_C.mp4": 'C',"videos/cam_D.mp4":'D'}
CurrentLight = ''

for video in ("videos/cam_A.mp4","videos/cam_B.mp4","videos/cam_C.mp4", "videos/cam_D.mp4"): # 4,1,2,3
# for video in ["videos/video15s.mp4"]:

    args["input"] = video

    # YOLOv3 configuration
    print("[INFO] loading model...")
    net = cv2.dnn.readNet(args["yolo"] + "/yolov3_608.weights", args["yolo"] + "/yolov3_608.cfg")
    print("[INFO] path to weights: ", args["yolo"] + "/yolov3_608.weights")
    print("[INFO] path to cfg: ", args["yolo"] + "/yolov3_608.cfg")

    with open(args["yolo"] + "/yolov3_608.names", 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # NN input image size
    inpWidth  = 608
    inpHeight = 608

    # Input video path
    print("[INFO] input directory: ", args["input"])

    # Reading the video
    # print("[INFO] opening video file...")
    print("[INFO] processing video file...")
    vs = cv2.VideoCapture(args["input"])
    # vs.set(cv2.CAP_PROP_POS_MSEC,3000)      # jump to 20 sec

    CurrentLight = FixLight[video]
    # print(FixLight[video],video,CurrentLight)

    # Input video size
    width = None
    height = None

    # Initializing tracking algorithm
    # maxDisappeared = the number of frames on which a car may disappear from out vision
    # maxDistance = max distance between the centroids of one car on two different frames
    car_ct = CentroidTracker()
    car_ct.maxDisappeared = 10
    truck_ct = CentroidTracker()
    truck_ct.maxDisappeared = 10

    # the list of trackers
    trackers = []
    # lists of objects we are going to track
    car_trackableObjects = {}
    truck_trackableObjects = {}

    # The number of frames in the video
    totalFrames = 0

    # The number of processed frame
    frame_number = 0

    # Those are dicts of "old" trackers that relate to the previous frame
    # They are compared to "new" trackers that relate to the current frame
    old_car_trackers = None
    old_truck_trackers = None

    # This array contains IDs of cars that are, probably, stopped
    stopped_car_IDs = []
    stopped_truck_IDs = []

    # Tn those dicts key = ID of a car that is, perhaps, stopped and value = amount of SECONDS that this car was not moving
    car_counting_frames = {}
    truck_counting_frames = {}

    # In those dicts key = ID of a car that is, perhaps, stopped and value is an array [date&time when car stopped, current date&time]
    car_counting_seconds = {}
    truck_counting_seconds = {}

    # This is in how many parts we separate the width of a frame to use it in detection later
    # parts = 80
    parts = 10

    # This is hot many frames a car should not move so we can say that it has actually stopped
    # If a video was shot in 60fps then 30 frames is equal to 0.5 second of standing on one place.
    # You can change this variable if you want
    frames_to_stop = 30

    ########################################################################################################################

    while True:
        frame_number += 1
        success_capture, frame = vs.read()
        if not success_capture:
            # print("=============================================")
            # print("ERROR! VIDEO NOT FOUND")
            # print("=============================================")
            break
            # stop if the end of the video is reached
        if frame is None:
            # print("=============================================")
            # print("The end of the video reached")
            # print("=============================================")
            break

        ## Bally modify ##
        # framespersecond = int(vs.get(cv2.CAP_PROP_FPS))
        # print("FPS = ", framespersecond)

        # print("\n=============================================")
        # print(f"FRAME {frame_number}")

        # change frame size to increase speed a bit
        frame = imutils.resize(frame, width=600)

        #change colors from RGB to BGR to work in dlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if width is None or height is None:
            height, width, channels = frame.shape

        # print(f"minimum distance is {width / parts}")

        # lists of bounding boxes
        car_rects = []
        truck_rects = []

        # every N frames (look at "skip-frames" argument) vehicles DETECTION takes place
        # then between those frames every vehicles is being TRACKED
        # that increases the speed significantly

        ## bally modify ##
        IdToClassList = {}

        if totalFrames % args["skip_frames"] == 0:
            # empty list of trackers
            trackers = []
            # list of classes numbers
            class_ids = []

            # pass the blob-model of the frame through the NN to get boxes of detected objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (inpWidth, inpHeight), (0, 0, 0), True, crop=False)
            # blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # analyze boxes list
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    if (class_id != 2) and (class_id != 7):  # if a car or a truck is detected - continue
                        continue
                    confidence = scores[class_id]
                    if confidence > args["confidence"]:
                        # box'es center coords
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        # width of the box
                        w = int(detection[2] * width)
                        # height of the box
                        h = int(detection[3] * height)

                        # if a box is too small (for example if a car is moving very close to the edge of a frame, then we do skip it
                        if h <= (height / 10):
                            continue

                        # coords of left upper and right lower connors of the box
                        x1 = int(center_x - w / 2)
                        y1 = int(center_y - h / 2)
                        x2 = x1 + w
                        y2 = y1 + h

                        # let's make a maximum distance of centroid tracker equal to the width of a box
                        truck_ct.maxDistance = w
                        car_ct.maxDistance = w

                        # draw a box and write a detected class above it
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                        cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # create a tracker for every car
                        tracker = dlib.correlation_tracker()
                        # create a dlib rectangle for a box
                        rect = dlib.rectangle(x1, y1, x2, y2)
                        # start tracking each box
                        tracker.start_track(rgb, rect)
                        # every tracker is placed into a list
                        trackers.append(tracker)
                        class_ids.append(class_id)

                        ## bally modify ##
                        IdToClassList[tracker] = class_id
                        
            ## bally modify ##
            # print("IdToClassList", IdToClassList)


            # print("-"*43)
            # print("Count Total Car     : ", class_ids.count(2))
            # print("Count Total Truck   : ", class_ids.count(7))
            # print("Count Total vehicle : ", class_ids.count(2) + class_ids.count(7))
            # print("-"*43)


        # if frame number is not N then we work with previously created list of trackers rather that boxes
        else:
            for tracker, class_id in zip(trackers, class_ids):
                # a car was detected on one frame and after that on other frames it's coords are constantly updating
                tracker.update(rgb)

                pos = tracker.get_position()

                # get box coords from each tracker
                x1 = int(pos.left())
                y1 = int(pos.top())
                x2 = int(pos.right())
                y2 = int(pos.bottom())

                # draw a box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                obj_class = CLASSES[class_id]

                if obj_class == "car":
                    car_rects.append((x1, y1, x2, y2))
                elif obj_class == "truck":
                    truck_rects.append((x1, y1, x2, y2))

        # those are the lists of cars that we tracked and centroids were moved to their new positions
        cars = car_ct.update(car_rects)
        trucks = truck_ct.update(truck_rects)

        '''
        At the very beginning a thought that it was enough just to check if centroid coords on N and N+1 frames are the same
        That would indicate that a car is standing still.
        BUT
        YOLOv3 works not perfectly and even if I can see that the car is not moving it's centroid is "shaking" on the video
        That means that it's coords won't be exactly the same of two different frames.
        So I decided to consider a car not moving if it's centroid coords are SLIGHTLY changing but are still not very different on separate frames
        So how it works
        1) If distance between car centroid's coords on 1 and 2 frames is shorter than a minimum (we can change it) than we put this centroid if a dictionary.
        2) Dictionary looks like this
            ID (key) -> Number of frames on which distance that I was talking about above is "kept"(value)
        3) If on the 3rd frame situation is the same (distance is small) than the number of frames increases
        4) If this continues for some time (or some amount of frames) than we can tell that the car is stopped
        5) If after decreasing the distance starts increasing than we can tell that the car is moving again
        '''

        # get the IDs of cars that are, perhaps, stopped
        if (old_car_trackers is not None):
            stopped_car_IDs = compare_trackers(old_car_trackers, cars, width, stopped_car_IDs)
        if (old_truck_trackers is not None):
            stopped_truck_IDs = compare_trackers(old_truck_trackers, trucks, width, stopped_truck_IDs)
            if stopped_car_IDs != []:
                for ID in stopped_car_IDs:
                    # Increasing the number of frames
                    if ID in car_counting_frames.keys():
                        car_counting_frames[ID] += 1
                    # Adding a new car ID
                    else:
                        car_counting_frames[ID] = 1
                # if any ID is IN car_counting_frames.keys() but it os NOT IN the stopped_car_IDs then we have to delete
                # from the dictionary as it means that the car is stopped and moving at the same time which is impossible
                for ID in car_counting_frames.copy().keys():
                    if ID not in stopped_car_IDs:
                        # print(f"{ID + 1} is in car_counting_frames but is not in stopped_car_IDs - delete it from car_counting_frames")
                        car_counting_frames.pop(ID)
            else:
                # If a list is empty it means that there are no cars to process
                car_counting_frames = {}
            # same thing for trucks (you can add your classed here)
            if stopped_truck_IDs != []:
                for ID in stopped_truck_IDs:
                    if ID in truck_counting_frames.keys():
                        truck_counting_frames[ID] += 1
                    else:
                        truck_counting_frames[ID] = 1

                for ID in truck_counting_frames.copy().keys():
                    if ID not in stopped_truck_IDs:
                        truck_counting_frames.pop(ID)
            else:
                truck_counting_frames = {}

        # print("\n")

        # some info on the screen (debug)
        # for k,v in car_counting_frames.items():
            # print(f"car {k+1} was standing for {v} frames")
        # for k,v in truck_counting_frames.items():
            # print(f"truck {k+1} was standing for {v} frames")
        
        ### Bally Modify ###
        # print("car_counting_frames : ", car_counting_frames) # Don't forget to add key + 1
        # print("truck_counting_frames : ", truck_counting_frames) # Don't forget to add key + 1

        # print("Total Standing Car : ", len(car_counting_frames))
        # print("Total Standing Truck : ", len(truck_counting_frames))
        # if len(list(car_counting_frames.keys())) != 0:
        #     print("Max Car Count", max(list(car_counting_frames.keys()))+1)
        # if len(list(truck_counting_frames.keys())) != 0:
        #     print("Max Truck Count", max(list(truck_counting_frames.keys()))+1)

        # those are the lists of cars that have been standing still long enough
        # they refresh EACH frame
        long_stopped_cars = find_stopped_cars(car_counting_frames, frames_to_stop)
        long_stopped_trucks = find_stopped_cars(truck_counting_frames, frames_to_stop)

        # now when we have a list of cars that are for sure stopped we can count how long (in seconds) they are not moving
        for ID in long_stopped_cars:
            if ID not in car_counting_seconds.keys():
                # if it is a new car then we pinpoint time when car stops
                start = time.asctime()
                # [0,0] will later be replaced
                car_counting_seconds[ID] = [0,0]
                car_counting_seconds[ID][0] = start
            else:
                # else if this car is already on the list then we add time to it's current time
                stop = time.asctime()
                car_counting_seconds[ID][1] = stop

        # do the same thing but for trucks
        for ID in long_stopped_trucks:
            if ID not in truck_counting_seconds.keys():
                # if it is a new car then we pinpoint time when car stops
                start = time.asctime()
                truck_counting_seconds[ID] = [0, 0]
                truck_counting_seconds[ID][0] = start
            else:
                # else if this car is already on the list then we add time to it's current time
                stop = time.asctime()
                truck_counting_seconds[ID][1] = stop


        # print("\n----RESULTS----:")
        # # show info about seconds in command line
        # for ID,[start, stop] in car_counting_seconds.items():
        #     print(f"car {ID + 1} was standing from: {start} to {stop}")
        # for ID, [start, stop] in truck_counting_seconds.items():
        #     print(f"truck {ID + 1} was standing from: {start} to {stop}")


        old_car_trackers = cars.copy()
        old_truck_trackers = trucks.copy()

        # draw centroids for cars and trucks
        car_centroids = draw_centroids(frame, cars, car_trackableObjects, long_stopped_cars, ClassStr = "car")

        # print("car centroid : ", car_centroids)

        truck_centroids = draw_centroids(frame, trucks, truck_trackableObjects, long_stopped_trucks, ClassStr = "truck")

        # print("truck centroid : ", truck_centroids)

        ## Bally modify ##
        TotalVehicle = car_centroids | truck_centroids

        for k,v in car_counting_frames.items():
            TotalVehicle[k+1].append(v)
        for k,v in truck_counting_frames.items():
            TotalVehicle[k+1].append(v)

        ## set number of standing frames that is stop car ##
        StandToStopFrames = 1000 # 1000 frames = 34 sec

        # Total_Vehicle_count      = len(TotalVehicle)
        # Total_Car                = sum(1 for i in TotalVehicle.values() if i[1] == "car")
        # Total_Truck              = sum(1 for i in TotalVehicle.values() if i[1] == "truck")
        # Total_Stop_car           = 0 # Start

        ## bally modify fix frame of stop car ##
        for k,v in TotalVehicle.items():
            if len(v) == 3:
                pass
            else:
                TotalVehicle[k].append(0)

        # Total_Stop_car   = sum(1 for i in TotalVehicle.values() if i[2] >= StandToStopFrames) # test stading frame >= 10 = stop car

        # print("All Vehicle List : ", TotalVehicle)
        # print("\n")
        
        # print(''' Total Vehicle          = {0}
        #           Total Car              = {1}
        #           Total Truck            = {2}
        #           Total Stop Vehicle     = {3}
        #       '''.format(Total_Vehicle_count, Total_Car, Total_Truck, Total_Stop_car))
        
        
        LongestDictY = {}
        LongestControid = {}
        ## remove stop car ##
        for k,v in TotalVehicle.items():
            if v[2] < StandToStopFrames: # filter only standing car < 1000 frames = 34 sec
                LongestDictY[k]    = v[0][1]
                LongestControid[k] = v
        # print("LongestDictY ", LongestDictY)
        

        ## get min y centroid ##
        Min_y_key = 0 # start
        Min_y_val = [] # start
        ## find min y centroids
        if len(LongestDictY) != 0:
            # Max_y = min(LongestDictY, key=lambda item:max(item[1]))
            Min_y_key = min(LongestDictY, key=LongestDictY.get)
            Min_y_val = TotalVehicle[Min_y_key]
        
        # print("LongestControid", LongestControid)
        # print("\n")
        # print("No. Car Longest Y : ", Min_y_key)
        # print("\n")
        # print("Longest Y : ", Min_y_val)

        ## find max no of car in lane ##
        XfromMinY = 0 # start

        if len(Min_y_val) != 0:
            XfromMinY = Min_y_val[0][0]

        # print("XfromMaxY", XfromMaxY)

        XdictFromlongestControid = {} # start
        for k,v in LongestControid.items():
            if v[0][0] in range(int(XfromMinY*0.93), int(XfromMinY*1.07)): # GAP = 7%
                XdictFromlongestControid[k] = v
        
        # print("XdictFromlongestControid", XdictFromlongestControid)

        ## final : Find No. of car in lane and vehicle type ##
        TotalStackVenicleInLane       = len(XdictFromlongestControid)
        TotalStackVenicleInLane_car   = sum(1 for i in XdictFromlongestControid.values() if i[1] == "car")
        TotalStackVenicleInLane_Truck = sum(1 for i in XdictFromlongestControid.values() if i[1] == "truck")

        # print("\n")
        # print("*"*50)
        # print(" Total Stack Vehicle In Lane = ", TotalStackVenicleInLane, "\n", "Total Stack Car In Lane = ", TotalStackVenicleInLane_car, "\n", "Total Stack Truck   In Lane = ", TotalStackVenicleInLane_Truck)
        # print("XdictFromlongestControid",XdictFromlongestControid)
        # print("*"*50)

        # show the frame
        cv2.putText(frame, video[7:12], (10,37), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF


        # press "q" to stop working
        if key == ord("q") or totalFrames == Total_run_Frame:

            ## Green Light Time ##
            ## Time Calculation From Obervation ##
            CarTimeSec   = 5
            TruckTimeSec = 14

            Green_Time = TotalStackVenicleInLane_car*CarTimeSec + TotalStackVenicleInLane_Truck*TruckTimeSec + 0.5*(TotalStackVenicleInLane-1)

            ## min max time ##
            if Green_Time   >= 120:
                Green_Time   = 120
            elif Green_Time <= 15:# 15 for criteria
                Green_Time   = 5 # 5 for demo
            else:
                pass # same time sec.
            
            Total_run_Frame = Green_Time*29 # 29 FPS


            print("/"*43)
            print("\n")
            print(" Total Stack Vehicle In Lane = ", TotalStackVenicleInLane, "\n", "Total Stack Car In Lane = ", TotalStackVenicleInLane_car, "\n", "Total Stack Truck   In Lane = ", TotalStackVenicleInLane_Truck)
            print("Green Time at Road {0} = {1} Sec.".format(CurrentLight,Green_Time))
            print("\n")
            print("/"*43)
            ######################################################
            ######################################################
            # j = 0
            # while True:
            #     print("CurrentLight :",CurrentLight)
            #     ser.write(str.encode(CurrentLight))
            #     print("Sent")
            #     time.sleep(1)
            #     j += 1
            #     if j == 3:
            #         break

            ######################################################

            print("[INFO] process finished by user")
            break

        # increase frame number
        totalFrames += 1

    # fps.stop()
    # print("[INFO] elasped time: {:.10f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.10f}".format(fps.fps()))

        # ### bally modify ##########################
        # if cv2.waitKey(100) == 27: # ESC to exit
        #     break
            
        # time.sleep(delay_time[j])
        # j+=1
        # if j%3 == 0:
        #     j=0

        # if cv2.waitKey(100) == 27: # ESC to exit
        #     break
        # ################### run 3 sec ##############

    # close everything
    cv2.destroyAllWindows()

    # Thanks for using my code, bud ;)

