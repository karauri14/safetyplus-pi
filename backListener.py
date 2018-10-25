import RPi.GPIO as GPIO

BACK_PIN = 12

#refuel listener
def isBack():
    if (GPIO.input(BACK_PIN) == GPIO.LOW):
        return True
    
    return False