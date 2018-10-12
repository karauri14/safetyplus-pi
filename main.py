import RPi.GPIO as GPIO
import cv2
import numpy as np

#my module
import signListener
import turnListener
import fuelListener
import parkingListener
import langSelector
import ui

KEY_ESC=27
    
def main():
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(turnListener.SLOW_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(fuelListener.FUEL_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(turnListener.L_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(turnListener.R_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(langSelector.SET_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(langSelector.SELECT_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(parkingListener.PARK_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    
    prev_string = ""
    
    lang = 'ko'
    
    video = cv2.VideoCapture(0)
    
    bg_origin = cv2.imread('./img/background.png')
    background = bg_origin.copy()
    
    images = ui.windowInit(lang)
    
    sign_count = {'STOP':0, 'SLOW':0, 'OVER':0}
    turn_state = (",")
    
    
    #main loop
    while True:
        
        k = cv2.waitKey(1)
        if k == KEY_ESC:
            cv2.destroyAllWindows()
            break
        
        state_string = ""
        turn_state = turnListener.listener(turn_state)
        state_string += turn_state
        state_string += signListener.listener(video, sign_count)
        
        if (GPIO.input(langSelector.SELECT_PIN) == GPIO.LOW):
            lang = langSelector.langSelect(lang)
            images = ui.windowInit(lang)
            background = bg_origin.copy()
            ui.makeWindow(background, images, state_string)
            
        if prev_string != state_string:
            prev_string = state_string
            background = bg_origin.copy()
            ui.makeWindow(background, images, state_string)
        
        is_parking = parkingListener.listener()
        if is_parking:
            break
        
        cv2.waitKey(1)
        
    cv2.imshow('drive', images['DOOR'])
    
    while True:
        if fuelListener.listener():
            ui.fuelWindow(bg_origin, images)
            cv2.waitKey(0)
            break
        
        k = cv2.waitKey(1)
        if k == KEY_ESC:
            cv2.destroyAllWindows()
            break
    
    video.release()        
    GPIO.cleanup()
    
if __name__ == "__main__":
    main()
