import RPi.GPIO as GPIO

FUEL_PIN = 26

#refuel listener
def listener():
    if (GPIO.input(FUEL_PIN) == GPIO.LOW):
        return ("fuel,")
    else :
        return (",")