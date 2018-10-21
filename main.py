import RPi.GPIO as GPIO
import cv2
import numpy as np

#my module
import signListener
import turnListener
import fuelListener
import textListener
import parkingListener
import langSelector
import ui

KEY_ESC=27
    
def main():
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(fuelListener.FUEL_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(turnListener.SLOW_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(turnListener.L_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(turnListener.R_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(langSelector.SET_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(langSelector.SELECT_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(parkingListener.PARK_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(textListener.BACK_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(textListener.DANGER_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)

    prevString = ""
    
    lang = 'ko'
    
    video = cv2.VideoCapture(0)
    
    bgOrigin = cv2.imread('./img/background.png')
    background = bgOrigin.copy()
    
    images = ui.windowInit(lang)
    
    signListener.init()
    signCount = {'STOP':0, 'SLOW':0, 'OVER':0}
    turnState = (",")
    
    
    #main loop
    while True:
        
        stateString = ""
        if parkingListener.isParking() != True:
            
            #main area
            turnState = turnListener.listener(turnState)
            stateString += turnState
            
            #text area
            stateString += textListener.listener()
            
            #sign area
            stateString += signListener.listener(video, signCount)
            
            #language select
            if (GPIO.input(langSelector.SELECT_PIN) == GPIO.LOW):
                lang = langSelector.langSelect(lang)
                images = ui.windowInit(lang)
                background = bgOrigin.copy()
                ui.makeWindow(background, images, stateString)

            elif prevString != stateString:
                prevString = stateString
                background = bgOrigin.copy()
                ui.makeWindow(background, images, stateString)
        
        #parking
        else :
            prevString = 'parking'
            cv2.imshow('drive', images['DOOR'])
            if fuelListener.isFuel():
				cv2.imshow('drive', images['REFUEL'])

        k = cv2.waitKey(1)
        if k == KEY_ESC:
            break
    
    cv2.destroyAllWindows()
    video.release()        
    GPIO.cleanup()
    
if __name__ == "__main__":
    main()
