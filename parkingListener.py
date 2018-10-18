import RPi.GPIO as GPIO

DANGER_PIN = 5
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

def isDanger():
    if (GPIO.input(DANGER_PIN) == GPIO.LOW):
        return (True)

    return (False)