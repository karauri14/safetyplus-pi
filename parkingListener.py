import RPi.GPIO as GPIO

PARK_PIN = 6

def isParking():
    if (GPIO.input(PARK_PIN) == GPIO.LOW):
        return (True)
    
    return (False)

