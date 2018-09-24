import cv2
import numpy as np

#sign listener
def matching_sign(image, temp_path):
    temp = cv2.imread(temp_path, 0)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gamma = 2.0
    look_up_table = np.ones((256, 1), dtype = 'uint8') * 0
    for i in range(256):
        look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    img = cv2.LUT(img, look_up_table)
    
    temp = cv2.resize(temp, (img.shape[1], img.shape[0]))
    temp = cv2.GaussianBlur(temp, (5, 5), 0)
    result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where(result >= threshold) 
    flag = False
    for top_left in zip(*loc[::-1]):
        flag = True
    
    return flag

def detect_contour(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    retval, bw_d = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(bw_d, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    retval, bw_l = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    image, contours2, hierarchy2 = cv2.findContours(bw_l, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contours.extend(contours2)
    
    is_stop = False
    
    for i in range(0, len(contours)):
        
        area = cv2.contourArea(contours[i])
        
        if area < 300 or 5000 < area:
          continue
        
        if len(contours[i]) > 0:
            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)
            if (h / w) < 0.9 and (h / w) > 0.8:
                is_stop = matching_sign(src[y:y + h, x:x + w], "./template/temp_stop.jpg")
    
    if (is_stop == True):
        return ("stop,,")
    else :
        return (",,")

def listener(video):
    is_read, frame = video.read()
    
    if frame is not None:
        return (detect_contour(frame))
