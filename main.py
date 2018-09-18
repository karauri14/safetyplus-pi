import obd
import RPi.GPIO as GPIO
import cv2
import numpy as np

KEY_ESC=27

FUEL_PIN = 26
L_PIN = 20
R_PIN = 21
#demo only
SWITCH_PIN = 16
##

MAX_LENGTH = 10

SIGN_WIDTH = 176
SIGN_HEIGHT = 160

#refuel listener
def refuelListener():
    if (GPIO.input(FUEL_PIN) == GPIO.LOW):
        return ("fuel,")
    else :
        return (",")
##

#turn listener
def speedLog(speed):
    
    #con = obd.OBD()
    for i in range(0, MAX_LENGTH - 1):
        speed[i] = speed[i + 1]

    speed[MAX_LENGTH - 1] = 60
    #speed[MAX_LENGTH - 1] = con.query(obd.commands.SPEED)

def speed_check(speed):
    for i in range(0, MAX_LENGTH):
        if (speed[i] < speed[i + 1]):
            return False

    return True
    
def turnListener(speed):
    
    speedLog(speed)
    if (GPIO.input(L_PIN) == GPIO.LOW or GPIO.input(R_PIN) == GPIO.LOW):
        flag = speed_check(speed)
        if (GPIO.input(SWITCH_PIN) == GPIO.LOW and flag == True):
            if (GPIO.input(L_PIN) == GPIO.LOW):
                return ("left,")
            elif (GPIO.input(R_PIN) == GPIO.LOW):
                return ("right,")
    
    return (",")
##

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
        return ("stop,,,")
    else :
        return (",,,")

def camera(video):
    is_read, frame = video.read()
    
    if frame is not None:
        cv2.imshow('video', frame)
        return (detect_contour(frame))
##

def windowInit():
    cv2.namedWindow('drive', cv2.WINDOW_AUTOSIZE)
    #cv2.setWindowProperty('drive', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    
    bg_origin = cv2.imread('./img/background.png')
    background = bg_origin
    
    stop_sign = cv2.imread('./img/ko/stop.png')
    stop_sign = cv2.resize(stop_sign, (SIGN_WIDTH, SIGN_HEIGHT))
    slow_sign = cv2.imread('./img/ko/slow.png')
    slow_sign = cv2.resize(slow_sign, (SIGN_WIDTH, SIGN_HEIGHT))
    over_sign = cv2.imread('./img/ko/overtaking.png')
    over_sign = cv2.resize(over_sign, (SIGN_WIDTH, SIGN_HEIGHT))
    
    pastePicture(background, stop_sign, 0, 0)
    pastePicture(background, slow_sign, 0, SIGN_HEIGHT)
    pastePicture(background, over_sign, 0, SIGN_HEIGHT * 2)
    cv2.imshow('drive', background)

def pastePicture(background, src, x, y):
    
    row, col, channel = src.shape
    
    graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(graySrc, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    roi = background[0 + y : row + y, 0 + x : col + x]
    
    bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    srcFg = cv2.bitwise_and(src, src, mask = mask)
    dst = cv2.add(bg, srcFg)
    background[0 + y : row + y, 0 + x : col + x] = dst
    
def main():
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(FUEL_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(L_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(R_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    
    speed = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 0]
    
    prev_string = ""
    
    video = cv2.VideoCapture(0)
    
    windowInit()
    
    while True:
        state_string = ""
        state_string += refuelListener()
        state_string += turnListener(speed)
        state_string += camera(video)
        
        if prev_string != state_string:
            prev_string = state_string
            print(state_string)
        
        k = cv2.waitKey(1)
        if k == KEY_ESC:
            cv2.destroyAllWindows()
            break

    video.release()        
    GPIO.cleanup()
    
if __name__ == "__main__":
    main()
