import obd
import RPi.GPIO as GPIO
import time
import threading

#speed log
MAX_LENGTH = 100
SLOW_SPEED = 20

#use GPIO pin
L_PIN = 20
R_PIN = 21

#demo only
#SLOW_PIN = 16
#SPEED = 60

#turn listener

con = obd.OBD()

speedRecode = []

def speedLog():
    speed = con.query(obd.commands.SPEED)
    if str(speed) != 'None':
        speedRecode.append(speed.value.magnitude)

    if len(speedRecode) >= MAX_LENGTH:
        del speedRecode[0]

def isSlow():

    for i in speedRecode:
        if (i <= SLOW_SPEED):
            return True
            
    return False

def listener(state):
    speedLog()
    if (GPIO.input(L_PIN) == GPIO.LOW):
        if state == 'left,':
            return ('left,')
        
        if isSlow():
            return ('left,')
        
    if (GPIO.input(R_PIN) == GPIO.LOW):
        if state == 'right,':
            return ('right,')
        
        if isSlow():
            return ('right,')
        
    return (',')
