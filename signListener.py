import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

IMPORT_DIR = "./savedModel"
DATASET_IMAGE_SIZE = 32

MIN_AREA = 100
MAX_AREA = 4000
MIN_RATIO = 0.8
MAX_RATIO = 1.0

SIGN = np.array(["Takeover Sign", "Negative","Slow Sign", "Stop Sign"])

#delay time (ms)
OVER_TIME = 10
STOP_TIME = 5
SLOW_TIME = 30

LOWER_RED1 = np.array([0,10,20])
UPPER_RED1 = np.array([20,255,255])
LOWER_RED2 = np.array([160,10,20])
UPPER_RED2 = np.array([180,255,255])
    
ops.reset_default_graph()
sess = tf.Session()
GRAPH_NAME = {'input_image':'','keep_prob':'','prediction':''}

def init():
    tf.saved_model.loader.load(sess, ["serve"], IMPORT_DIR)
    graph = tf.get_default_graph()
    GRAPH_NAME['input_image'] = graph.get_tensor_by_name('x_input:0')
    GRAPH_NAME['keep_prob'] = graph.get_tensor_by_name('keep_prob:0')
    GRAPH_NAME['prediction'] = graph.get_tensor_by_name('prediction:0')

#sign listener
def classification(ROI):
    
    shape = ROI.shape

    ROI_arr = [cv2.resize(ROI, (DATASET_IMAGE_SIZE,DATASET_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)]
    ROI_arr = np.array(ROI_arr)
    
    ROI_arr = (ROI_arr-ROI_arr.mean())/(ROI_arr.max()-ROI_arr.min())
    
    pred = sess.run(GRAPH_NAME['prediction'], feed_dict={GRAPH_NAME['input_image']:ROI_arr, GRAPH_NAME['keep_prob']:1.0})
    label = np.argmax(pred, 1)
    predictLabel = SIGN[int(label)]
     
    return(predictLabel)

def findContourUsingRedFilter(src):
    redSegment = redMask(src)
    image, contours, hierarchy = cv2.findContours(redSegment, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def redMask(src):
    hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv,LOWER_RED1,UPPER_RED1)
    mask2 = cv2.inRange(hsv,LOWER_RED2,UPPER_RED2)

    mask = cv2.bitwise_or(mask1, mask2)

    maskNot = cv2.bitwise_not(mask)

    return maskNot

def detectContour(src, signCount):

    contours = findContourUsingRedFilter(src)
    
    for cnt in contours:
        #area(float) of each countour(list)
        area = cv2.contourArea(cnt)

        #if the size of area less than 100 || more than 4000, just ignore
        if area < MIN_AREA or MAX_AREA < area:
          continue
        
        #create bounding box(x,y,w,h) around countours
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(h) / float(w)

        if ratio < MAX_RATIO and ratio > MIN_RATIO:
            ROI = src[y:y + h, x:x + w]
            
            #"Takeover Sign", "Negative","Slow Sign", "Stop Sign"
            sign = classification(ROI)
            if sign == 'Takeover Sign':
                signCount['OVER'] = OVER_TIME
            elif sign == 'Slow Sign':
                signCount['SLOW'] = SLOW_TIME
            elif sign == 'Stop Sign':
                signCount['STOP'] = STOP_TIME
      
    return (makeStateString(signCount))

def makeStateString(signCount):
    
    string = ''
    '''
    for key in signCount:
        if signCount[key] != 0:
            signCount[key] -= 1
            string += (key.lower() + ',')
        else :
            string += ','

    string = string[:-1]
    '''
    
    if signCount['STOP'] != 0:
        signCount['STOP'] -= 1
        string += 'stop,'
    else :
        string += ','
        
    if signCount['SLOW'] != 0:
        signCount['SLOW'] -= 1
        string += 'slow,'
    else :
        string += ','
        
    if signCount['OVER'] != 0:
        signCount['OVER'] -= 1
        string += 'over'
    else :
        string += ''
        
    return (string)

def listener(video, signCount):
    isRead, frame = video.read()
    
    if frame is not None:
        return (detectContour(frame, signCount))
