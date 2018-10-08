import obd
import RPi.GPIO as GPIO
import time
import threading

#speed log
MAX_LENGTH = 100
MARGIN = 5

#use GPIO pin
L_PIN = 20
R_PIN = 21

#demo only
SLOW_PIN = 16
SPEED = 60

#turn listener
'''
con = obd.OBD()
'''
speedThread = threading.Thread(target = speedLog)
firstSpeed = 0
speed = []
    
def speedLog():
    
    speed[:] = []
    #demo only
    for i in range(0, MAX_LENGTH):
        if (GPIO.input(SLOW_PIN) == GPIO.LOW):
            speed.append = firstSpeed - MARGIN
        
        else :
            speed.append = SPEED
        
        time.sleep(0.01)
    
    '''
    for i in range(0, MAX_LENGTH):
        speed[i] = con.query(obd.commands.SPEED)
        time.sleep(0.01)
    '''

def isSlow():
    if speedThread.is_alive() == False:

        #demo only
        firstSpeed = SPEED

        '''
        firstSpeed = con.query(obd.commands.SPEED)
        '''
        speedThread.start()

    for i in speed:
        if (i <= firstSpeed - MARGIN):
            return True
            
    return False

def listener():

    if (GPIO.input(L_PIN) == GPIO.LOW):
        if isSlow():
            return ("left,")
        
    if (GPIO.input(R_PIN) == GPIO.LOW):
        if isSlow():
            return ("right,")
    
    return (",")