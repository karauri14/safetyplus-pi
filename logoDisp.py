import cv2

WAIT_TIME = 1 
STEP = 0.04

def fadeIn(logo, bg):
    i = 0
    while i <= 1:
        res = cv2.addWeighted(logo, i, bg, 1 - i, 0)
        i += STEP
        cv2.imshow('logo', res)
        cv2.waitKey(WAIT_TIME)

def fadeOut(logo, bg):
    i = 0
    while i <= 1:
        res = cv2.addWeighted(logo, 1 - i, bg, i, 0)
        i += STEP
        cv2.imshow('logo', res)
        cv2.waitKey(WAIT_TIME)

def main():
    #cv2.namedWindow('logo', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('logo', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('logo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logo = cv2.imread('./img/logo.png')
    bg = cv2.imread('./img/background.png')
    
    cv2.imshow('logo', bg)
    cv2.waitKey(1)
    fadeIn(logo, bg)
    
    cv2.imshow('logo', logo)
    cv2.waitKey(3000)
    
    fadeOut(logo, bg)
    
    cv2.imshow('logo', bg)
    cv2.waitKey(1)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
