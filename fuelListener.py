import RPi.GPIO as GPIO

FUEL_PIN = 26

#refuel listener
def isFuel():
    if (GPIO.input(FUEL_PIN) == GPIO.LOW):
        return True
    
    return False
