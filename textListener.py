import RPi.GPIO as GPIO

DANGER_PIN = 5
BACK_PIN = 12

def isBack():
    if (GPIO.input(BACK_PIN) == GPIO.LOW):
        return (True)

    return (False)

def isDanger():
    if (GPIO.input(DANGER_PIN) == GPIO.LOW):
        return (True)

    return (False)

def listener():	
	if isBack():
		return ('back,')
				
	elif isDanger():
		return ('danger,')
		
	return (',')
