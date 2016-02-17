'''
Exercise 2 
 
Authors:
    Toyberman Maxim 307097451
    Shvartsman Yevgniy  310360961

Date:14-Apr-15
'''
import numpy as np
import cv2
from scipy import signal
import threading

condition=threading.Condition()
frames=[]
result=None
frame=None

def captureVideo():
    '''
    this function captures the video from camera in a given interval
    '''
    global frames,frame
    
    frames=[]
    cap = cv2.VideoCapture(0)
    #creating an event that will wakeup every 0.25 seconds
    stopped=threading.Event()

    while(not stopped.wait(0.25)):
        # Capture frame-by-frame
        ret,frame = cap.read()
        #sometimes first frame is null due to camera warmup
        if not ret:
            continue  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        frames.append(gray)
        with condition:# condition variable notify on append of frame
            condition.notifyAll()
              
    # When everything done, release the capture    
    cap.release()
    cv2.destroyAllWindows()
    
    

def makeGaussKernel(size_x,size_y):
    '''
        This function creates gaussian kernel 
    '''
    kernel=np.zeros((size_x,size_y))
    
    for x in range(0,size_x):
        for y in range(0,size_y):
            kernel[x][y]=np.exp(-(x**2/float(size_x)+(y**2)/float(size_y)))
    
    return kernel

kernel=makeGaussKernel(3,3)

def convolution2D(frames,kernel):
    '''
    creating a list of convoluted frames 
    '''
    convolved=[]
    for frame in frames:
        convolved.append(signal.convolve2d(frame,kernel,mode='same')) 
    return convolved

def weightedMean(frames):
    '''
    calculating the average frame
    '''
    height,width=frames[0].shape
    final=np.zeros((height,width))

    weight_sum=0
    for i in range(0,len(frames)):
        weight_sum+=(i+1)
        frames[i]*=(i+1)
        frame=frames[i]/1.0
        final=cv2.add(final,frame)

    return (final/float(weight_sum))


def linear_Normalization(lst, i):
    '''
    linear noramlization of the frame  (I-Min)*(new_Max-new_Min)/(Max-Min) +New_min
    '''
    newMax=255.0
    Min = lst[i].min()
    Max = lst[i].max()
    lst[i] -= Min
    try:
        
        lst[i] *= newMax / (Max - Min)
        
    except ZeroDivisionError:
        print('')
    
    #return np.uint8(lst)

def normalize(frames):
    '''
    looping over the frames and applying the linear_normalization
    '''
    new_lst=[]
    
    for i in range(0,len(frames)):
        new_lst.append(np.array(frames[i]))
        linear_Normalization(new_lst, i)
    return new_lst


def trackMovement():
    '''
    tracking movement EX 1 
    '''
    lst=[]
    global frames,result
    while True:
        with condition:
            condition.wait()
       
        convolved=convolution2D(frames,kernel)
        normalized_convolved=normalize(convolved)
        weighthed=weightedMean(normalized_convolved)
        linear_Normalization(weighthed, 0)
        result=np.uint8(weighthed)-np.asarray(normalized_convolved[-1]) ##convolved ch 
        
        lst.append(result) 
        cv2.imshow('Black_And_White',result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def draw_Contours():
    '''
    Drawing contours on the frame,based on tracked_movement
    '''
    global result,frame
    while(True):
        #waiting to be notifed for frame insertion
        with condition:
            condition.wait()
                
        res=np.array(result,dtype='float32')
        
        _,thresh = cv2.threshold(res,5,255,cv2.THRESH_BINARY)
        
        contours,_ = cv2.findContours(np.uint8(thresh),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255,255,255), 3)
        
        cv2.imshow('color',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
def clear_Frames():
    global frames
    frames=[]
    threading.Timer(1, clear_Frames).start()
 
try:
    threading.Thread(target=captureVideo).start()
    threading.Thread(target=trackMovement,).start()
    threading.Thread(target=clear_Frames,).start()
    threading.Thread(target=draw_Contours,).start()
except:
    print ("Error: unable to start thread")


        