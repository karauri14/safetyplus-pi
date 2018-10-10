import RPi.GPIO as GPIO
import time
import cv2

#my module
import ui

SET_PIN = 19
SELECT_PIN = 13

#wait 10s
CANCEL_TIME = 10 * 100

def langSelect(lang):
    
    bg_origin = cv2.imread('./img/lang.png')
    frame = cv2.imread('./img/frame.png')
    
    changed_lang = lang
    count = 0
    
    while True:
        count += 1
        if (count > CANCEL_TIME):
            break
        if (GPIO.input(SELECT_PIN) == GPIO.LOW):
            if changed_lang == 'en':
                changed_lang = 'zh-tw'
            elif changed_lang == 'zh-tw':
                changed_lang = 'zh-cn'
            elif changed_lang == 'zh-cn':
                changed_lang = 'ko'
            elif changed_lang == 'ko':
                changed_lang = 'en'
        
        ui.langSelectWindow(bg_origin, frame, changed_lang)
        
        if (GPIO.input(SET_PIN) == GPIO.LOW):
            return changed_lang
        
        time.sleep(0.01)
        
    return lang
