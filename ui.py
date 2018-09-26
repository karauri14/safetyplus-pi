import cv2
import numpy as np
import pygame.mixer
import time
import threading

MAX_WIDTH = 800
MAX_HEIGHT = 480
SIGN_HEIGHT = MAX_HEIGHT / 3
SIGN_WIDTH = SIGN_WIDTH
TEXT_WIDTH = MAX_WIDTH - SIGN_WIDTH
TEXT_HEIGHT = 50
GUIDE_WIDTH = MAX_WIDTH - SIGN_WIDTH
GUIDE_HEIGHT = MAX_HEIGHT - TEXT_HEIGHT

#call when system wake up
def logoDisp():
    cv2.namedWindow('logo', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('logo', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('logo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logo = cv2.imread('./img/logo.png')
    cv2.imshow('logo', logo)

    for i in range(0, 5):
        cv2.waitKey(10)
    cv2.destroyWindow('logo')

def soundPlay():
    pygame.mixer.music.play(1)
    time.sleep(10)
    pygame.mixer.music.stop()

def windowInit(lang):
    cv2.namedWindow('drive', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('drive', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('drive', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    pygame.init()
    pygame.mixer.music.load(lang + '/voice/turn.mp3')

    images = {}
    images['OIL'] = cv2.imread('./img/oil/regular.png')
    images['OIL'] = cv2.resize(images['OIL'], (GUIDE_WIDTH, GUIDE_HEIGHT))
    images['OILTEXT'] = cv2.imread(lang + '/img/refuel.ping')
    images['LEFT'] = cv2.imread('./img/turn_left.png')
    images['LEFT'] = cv2.resize(images['LEFT'], (GUIDE_WIDTH, GUIDE_HEIGHT))
    images['RIGHT'] = cv2.imread('./img/turn_right.png')
    images['RIGHT'] = cv2.resize(images['RIGHT'], (GUIDE_WIDTH, GUIDE_HEIGHT))
    images['STOP'] = cv2.imread(lang + '/img/stop.png')
    images['STOP'] = cv2.resize(images['STOP'], (SIGN_WIDTH, SIGN_HEIGHT))
    images['SLOW'] = cv2.imread(lang + '/img/slow.png')
    images['SLOW'] = cv2.resize(images['SLOW'], (SIGN_WIDTH, SIGN_HEIGHT))
    images['OVER'] = cv2.imread(lang + '/img/overtaking.png')
    images['OVER'] = cv2.resize(images['OVER'], (SIGN_WIDTH, SIGN_HEIGHT))
    images['OVERTEXT'] = cv2.imread(lang + '/img/over.png')
    images['OVERTEXT'] = cv2.resize(images['OVERTEXT'], (TEXT_WIDTH, TEXT_HEIGHT))
    images['ATTENTION'] = cv2.imread(lang + '/img/attention.png')
    images['ATTENTION'] = cv2.resize(images['ATTENTION'], (GUIDE_WIDTH, GUIDE_HEIGHT))
    images['ACCIDENT'] = cv2.imread(lang + '/img/accident.png')
    images['ACCIDENT'] = cv2.resize(images['ACCUDENT'], (TEXT_WIDTH, TEXT_HEIGHT))

    return images

def pastePicture(background, src, x, y):
    
    row, col, channel = src.shape
    
    graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(graySrc, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    roi = background[0 + y : row + y, 0 + x : col + x]
    
    bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    srcFg = cv2.bitwise_and(src, src, mask = mask)
    dst = cv2.add(bg, srcFg)
    background[0 + y : row + y, 0 + x : col + x] = dst

def makeWindow(base, images, state_string):
    fuel, turn, stop, slow, over = state_string.split(',')
    
    #main area
    if fuel == 'fuel':
        pastePicture(base, images['OIL'], SIGN_WIDTH, TEXT_HEIGHT)
        
    elif turn == 'left':
        pastePicture(base, images['LEFT'], SIGN_WIDTH, TEXT_HEIGHT)
        soundThread = threading.Thread(target = soundPlay)
        soundThread.start()
        
    elif turn == 'right':
        pastePicture(base, images['RIGHT'], SIGN_WIDTH, TEXT_HEIGHT)
        soundThread = threading.Thread(target = soundPlay)
        soundThread.start()
    
    else :
        pastePicture(base, images['ATTENTION'], SIGN_WIDTH, TEXT_HEIGHT)
    
    #sign area
    if stop == 'stop':
        pastePicture(base, images['STOP'], 0, 0)
    
    if slow == 'over':
        pastePicture(base, images['OVER'], 0, SIGN_HEIGHT)
    
    if over == 'slow':
        pastePicture(base, images['SLOW'], 0, SIGN_HEIGHT * 2)
    
    cv2.imshow('drive', base)
