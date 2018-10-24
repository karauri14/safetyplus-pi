import cv2
import numpy as np

# Define range of color in HSV yellow
lower_color = np.array([10, 30, 180])
upper_color = np.array([20,  255, 255])

count = {'LANE':0}

MAX_COUNT = 10

MIN_AREA = 2.5
MAX_AREA = 30

def isNotOver(frame):
	frame = framee[frame.shape[1] / 2:,frame.shape[0] / 2:]
    #resize_frame = cv2.rectangle(frame, (0,0), (int(frame.shape[1]/2)+15, int(frame.shape[0])), (0,0,0), thickness=-1)
    #resize_frame = cv2.rectangle(resize_frame, (int(resize_frame.shape[1]/2), 0), (int(resize_frame.shape[1]), int(resize_frame.shape[0]/2)), (0,0,0), thickness=-1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only yellow colors
    img_mask = cv2.inRange(hsv, lower_color, upper_color)

    image, contours, hierarchy = cv2.findContours(img_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range (0, len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        if area < MIN_AREA or MAX_AREA < area:
            continue

        if MIN_AREA < area and area < MAX_AREA :
            count['LANE'] += 1

        else :
            count['LANE'] = 0
            return False

        if  lane_cnt == MAX_COUNT:
            count['LANE'] = 0
            return True
