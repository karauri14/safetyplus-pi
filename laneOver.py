import cv2
import numpy as np

def nothing (x):
    pass

def laneover(video, pen):
    # Take each frame
    _, frame = video.read()
    
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, pen)
    # 読み込んだ動画(画像)に膨張処理をしている
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, pen)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(closing, cv2.COLOR_BGR2HSV)

    # Define range of color in HSV H(色) S(色の濃淡) V(明暗)
    lower_color = np.array([15, 40, 150]) 
    upper_color = np.array([25, 255, 255])
    
    # Threshold the HSV image to get only yellow colors
    img_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Bitwise-AND mask and original iamge
    #res = cv2.bitwise_and(closing, closing, mask = img_mask)
    
    try :
        image, contours, hierarchy = cv2.findContours(img_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        for i in range (0, len(contours)):
            #ここでモーメント(各要素,座標など)求めている
            cnt = contours[i]
            M = cv2.moments(cnt)
            #領域が占める面積をモーメントから読み込む
            area = cv2.contourArea(cnt)

            if area < 2000 or 8000 < area:
                continue
            
            print(area)
            x,y,w,h = cv2.boundingRect(img_mask)
            aspect_ratio = float(h)/w
            cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(res, ((int)(M['m10']/ M['m00']), (int)(M['m01']/ M['m00'])), 3, (0, 255, 255), -1)
            
            if 1 < aspect_ratio:
                return "over"
            
            else :
                return ""

    except :
        pass
    
    '''
    cv2.imshow('closing', closing)
    cv2.imshow('res', res)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
    cap.release()
    cv2.destroyAllWindows()
    '''
