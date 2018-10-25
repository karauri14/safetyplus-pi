import RPi.GPIO as GPIO

DANGER_PIN = 5

def isDanger():
    if (GPIO.input(DANGER_PIN) == GPIO.LOW):
        return (True)

    return (False)

def listener():			
	if isDanger():
		return ('danger,')
		
	return (',')
