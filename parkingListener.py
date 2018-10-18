import RPi.GPIO as GPIO

PARK_PIN = 6
BACK_PIN = 12

def isParking():
    if (GPIO.input(PARK_PIN) == GPIO.LOW):
        return (True)
    
    return (False)

def isBack():
    if (GPIO.input(BACK_PIN) == GPIO.LOW):
        return (True)

    return (False)
