import cv2
import numpy as np

# Define range of color in HSV yellow
lower_color = np.array([10, 30, 180])
upper_color = np.array([20,  255, 255])

def isNotOver(frame):

    resize_frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)))
    resize_frame = cv2.rectangle(resize_frame, (0,0), (int(resize_frame.shape[1]/2)+15, int(resize_frame.shape[0])), (0,0,0), thickness=-1)
    resize_frame = cv2.rectangle(resize_frame, (int(resize_frame.shape[1]/2), 0), (int(resize_frame.shape[1]), int(resize_frame.shape[0]/2)), (0,0,0), thickness=-1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only yellow colors
    img_mask = cv2.inRange(hsv, lower_color, upper_color)

    image, contours, hierarchy = cv2.findContours(img_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
    for i in range (0, len(contours)):
        #calc moment
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        if area < 2.5 or 30 < area:
            continue

        if 2.5 < area and area < 30 :
            lane_cnt = lane_cnt  + 1;

        else :
            lane_cnt = 0;
            return False

        if  lane_cnt == 10:
            lane_cnt = 0;
            return True
