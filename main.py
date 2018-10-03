import RPi.GPIO as GPIO
import cv2
import numpy as np

#my module
import signListener
import turnListener
import fuelListener
import langSelector
import ui

KEY_ESC=27
    
def main():
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(turnListener.SWITCH_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(fuelListener.FUEL_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(turnListener.L_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(turnListener.R_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(langSelector.SET_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(langSelector.SELECT_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    
    prev_string = ""
    
    lang = 'ko'

    turnListener.init()
    
    video = cv2.VideoCapture(0)
    
    
    bg_origin = cv2.imread('./img/background.png')
    background = bg_origin.copy()
    
    images = ui.windowInit(lang)

    while True:
        
        if (GPIO.input(langSelector.SELECT_PIN) == GPIO.LOW):
            lang = langSelector.langSelect(lang)
            images = ui.windowInit(lang)
            
        state_string = ""
        state_string += fuelListener.listener()
        state_string += turnListener.listener()
        state_string += signListener.listener(video)
            
        if prev_string != state_string:
            prev_string = state_string
            background = bg_origin.copy()
            ui.makeWindow(background, images, state_string)
        
        k = cv2.waitKey(1)
        if k == KEY_ESC:
            cv2.destroyAllWindows()
            break

    video.release()        
    GPIO.cleanup()
    
if __name__ == "__main__":
    main()
