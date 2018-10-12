import obd
import RPi.GPIO as GPIO
import time
import threading

#speed log
MAX_LENGTH = 100
MARGIN = 5
INF = 1000

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
speedRecode = [INF] * MAX_LENGTH

def speedLog():
    
    #demo only
    for i in range(0, MAX_LENGTH):
        if (GPIO.input(SLOW_PIN) == GPIO.LOW):
            speedRecode[i] = SPEED - MARGIN
        
        else :
            speedRecode[i] = SPEED
        
        time.sleep(0.01)
    
    '''
    for i in range(0, MAX_LENGTH):
        speed[i] = con.query(obd.commands.SPEED)
        time.sleep(0.01)
    '''

def isSlow():
    
    speedThread = threading.Thread(target = speedLog)
    if speedThread.is_alive() == False:
        #demo only
        firstSpeed = SPEED
        '''
        firstSpeed = con.query(obd.commands.SPEED)
        '''
        speedThread.start()
    for i in speedRecode:
        if (i <= (firstSpeed - MARGIN)):
            return True
            
    return False

def listener(state):
    
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
