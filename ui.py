import cv2
import numpy as np
import pygame.mixer
import time
import threading

MAX_WIDTH = 800
MAX_HEIGHT = 480
SIGN_HEIGHT = (int)(MAX_HEIGHT / 3)
SIGN_WIDTH = SIGN_HEIGHT
GUIDE_WIDTH = MAX_WIDTH - SIGN_WIDTH
GUIDE_HEIGHT = MAX_HEIGHT
OIL_TEXT_WIDTH = 200
OIL_TEXT_HEIGHT = (int)(OIL_TEXT_WIDTH / 2)
OIL_TEXT_MARGIN = 100

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
    images['OIL'] = cv2.resize(images['OIL'], (MAX_WIDTH, MAX_HEIGHT))
    images['OILTEXT'] = cv2.imread(lang + '/img/refuel.png')
    images['OILTEXT'] = cv2.resize(images['OILTEXT'], (OIL_TEXT_WIDTH, OIL_TEXT_HEIGHT))
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
    images['ATTENTION'] = cv2.imread(lang + '/img/attention.png')
    images['ATTENTION'] = cv2.resize(images['ATTENTION'], (GUIDE_WIDTH, GUIDE_HEIGHT))

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
    turn, stop, slow, over = state_string.split(',')
    
    #main area    
    if turn == 'left':
        pastePicture(base, images['LEFT'], SIGN_WIDTH, 0)
        soundThread = threading.Thread(target = soundPlay)
        soundThread.start()
        
    elif turn == 'right':
        pastePicture(base, images['RIGHT'], SIGN_WIDTH, 0)
        soundThread = threading.Thread(target = soundPlay)
        soundThread.start()
    
    else :
        pastePicture(base, images['ATTENTION'], SIGN_WIDTH, 0)
    
    #sign area
    if stop == 'stop':
        pastePicture(base, images['STOP'], 0, 0)
    
    if over == 'over':
        pastePicture(base, images['OVER'], 0, SIGN_HEIGHT)
    
    if slow == 'slow':
        pastePicture(base, images['SLOW'], 0, SIGN_HEIGHT * 2)
    
    cv2.imshow('drive', base)

def fuelWindow(base, images):
    pastePicture(base, images['OIL'], 0, 0)
    pastePicture(base, images['OILTEXT'], MAX_WIDTH - OIL_TEXT_WIDTH - OIL_TEXT_MARGIN, OIL_TEXT_MARGIN)
    
    cv2.imshow('drive', base)