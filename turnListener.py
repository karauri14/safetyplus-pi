import obd
import RPi.GPIO as GPIO

#speed length
MAX_LENGTH = 10

#use GPIO pin
L_PIN = 20
R_PIN = 21

#demo only
SWITCH_PIN = 16

speed = {}

#turn listener
def init():
    for i in range(0, MAX_LENGTH):
        speed[i] = 1000
    speed[MAX_LENGTH] = 0
    
def speedLog():
    
    #con = obd.OBD()
    for i in range(0, MAX_LENGTH - 1):
        speed[i] = speed[i + 1]

    speed[MAX_LENGTH - 1] = 60
    #speed[MAX_LENGTH - 1] = con.query(obd.commands.SPEED)

def speed_check():
    for i in range(0, MAX_LENGTH):
        if (speed[i] < speed[i + 1]):
            return False

    return True
    
def listener():
    
    speedLog()
    if (GPIO.input(L_PIN) == GPIO.LOW or GPIO.input(R_PIN) == GPIO.LOW):
        flag = speed_check()
        if (GPIO.input(SWITCH_PIN) == GPIO.LOW and flag == True):
            if (GPIO.input(L_PIN) == GPIO.LOW):
                return ("left,")
            elif (GPIO.input(R_PIN) == GPIO.LOW):
                return ("right,")
    
    return (",")