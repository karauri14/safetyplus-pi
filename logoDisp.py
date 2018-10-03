import cv2

def main():
    cv2.namedWindow('logo', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('logo', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('logo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logo = cv2.imread('./img/logo.png')
    cv2.imshow('logo', logo)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
