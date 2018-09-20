import cv2
import numpy as np
import pygame.mixer
import time
import threading

SIGN_WIDTH = 176
SIGN_HEIGHT = 160
TEXT_WIDTH = 800 - SIGN_WIDTH
TEXT_HEIGHT = 105
GUIDE_WIDTH = 800 - SIGN_WIDTH
GUIDE_HEIGHT = 480 - TEXT_HEIGHT

def soundPlay():
    pygame.mixer.music.play()

def windowInit(lang):
    cv2.namedWindow('drive', cv2.WINDOW_AUTOSIZE)
    #cv2.setWindowProperty('drive', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    pygame.mixer.init()
    pygame.mixer.music.load(lang + '/voice/turn.mp3')
    
    images = {}
    images['OIL'] = cv2.imread('./img/oil/regular.png')
    images['OIL'] = cv2.resize(images['OIL'], (GUIDE_WIDTH, GUIDE_HEIGHT))
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

def makeWindow(background, images, state_string):
    fuel, turn, stop, slow, over = state_string.split(',')
    
    #main area
    if fuel == 'fuel':
        pastePicture(background, images['OIL'], SIGN_WIDTH, TEXT_HEIGHT)
        
    elif turn == 'left':
        pastePicture(background, images['LEFT'], SIGN_WIDTH, TEXT_HEIGHT)
        soundThread = threading.Thread(target = soundPlay)
        soundThread.start()
        
    elif turn == 'right':
        pastePicture(background, images['RIGHT'], SIGN_WIDTH, TEXT_HEIGHT)
        soundThread = threading.Thread(target = soundPlay)
        soundThread.start()
    
    else :
        pastePicture(background, images['ATTENTION'], SIGN_WIDTH, TEXT_HEIGHT)
    
    #sign area
    if stop == 'stop':
        pastePicture(background, images['STOP'], 0, 0)
    
    if slow == 'slow':
        pastePicture(background, images['SLOW'], 0, SIGN_HEIGHT)
    
    if over == 'over':
        pastePicture(background, images['OVER'], 0, SIGN_HEIGHT * 2)
    
    cv2.imshow('drive', background)
