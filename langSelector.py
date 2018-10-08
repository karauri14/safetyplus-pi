import RPi.GPIO as GPIO
import time

SET_PIN = 19
SELECT_PIN = 13

#wait 10s
CANCEL_TIME = 10 * 100

def langSelect(lang):
    changed_lang = lang
    count = 0
    
    while True:
        count += 1
        if (count > CANCEL_TIME):
            break
        if (GPIO.input(SELECT_PIN) == GPIO.LOW):
            if changed_lang == 'ko':
                changed_lang = 'tw'
            elif changed_lang == 'tw':
                changed_lang = 'en'
            elif changed_lang == 'en':
                changed_lang = 'ko'
        
        if (GPIO.input(SET_PIN) == GPIO.LOW):
            return changed_lang
        
        time.sleep(0.01)
        
    return lang
