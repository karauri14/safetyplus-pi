import cv2
import numpy as np

def isOver(frame, pen):
    
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, pen)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, pen)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(closing, cv2.COLOR_BGR2HSV)

    # Define range of color in HSV
    lower_color = np.array([15, 40, 150]) 
    upper_color = np.array([25, 255, 255])
    
    # Threshold the HSV image to get only yellow colors
    img_mask = cv2.inRange(hsv, lower_color, upper_color)
    
    image, contours, hierarchy = cv2.findContours(img_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
    for i in range (0, len(contours)):
        #calc moment
        cnt = contours[i]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)

        if area < 2000 or 8000 < area:
            continue
        
        x,y,w,h = cv2.boundingRect(img_mask)
        aspect_ratio = float(h)/w
        
        if 1 < aspect_ratio:
            return True
        
        else :
            return False
    
