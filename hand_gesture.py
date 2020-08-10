import cv2
import numpy as np
import copy
import math
import pyautogui
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
threshold1=2
scr_thresh=100
# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works
xy=0
prev=0
def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt,far
    return False, 0,0
sx,sy=0,0
#movement of the mouse on the screen
def movement(check,point,prev,thresh):
    global sx,sy
    curx,cury=pyautogui.position()
    sx=point[0]-prev[0]
    sy=point[1]-prev[1]
    pyautogui.moveTo(curx+sx*thresh,cury+sy*thresh)
    return check

def show(thr):
    print("Mouse sensitivity has been changed to : "+str(thr))

def speed(thr):
    print("Scrolling speed has been changed to : "+str(thr))

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
#For varying the sensitivity of the mouse and the speed of scrolling
cv2.namedWindow('sensitivity')
cv2.createTrackbar('Mouse sensitivity','sensitivity',threshold1,4,show)
cv2.createTrackbar('Scrolling speed','sensitivity',scr_thresh,200,speed)
check=1
while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)


        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
            #drawing the contours on the final image
            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt,point = calculateFingers(res,drawing)
            thresh=cv2.getTrackbarPos('Mouse sensitivity','sensitivity')
            thresh2=cv2.getTrackbarPos('Scrolling speed','sensitivity')
            #If two fingers are seen 
            #Move your two fingers to move the mouse on the screen
            if cnt==1:
                check=check+1
                if xy==0:
                    prev=point
                    xy=1
                if point!=0:
                    check = movement(check,point,prev,thresh)
                    prev=point
            else:
                check=0
                xy=0
            #If three fingers are seen
            #Used for scrolling up and down 
            if cnt==2:
                cv2.rectangle(drawing,(5,5),(155,379),(255,0,0),-1)
                cv2.putText(drawing,"Up",(70,180),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                cv2.rectangle(drawing,(165,5),(315,379),(0,0,255),-1)
                cv2.putText(drawing,"Down",(220,180),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
                if point[0]>=5 and point[0]<=165:
                    pyautogui.scroll(thresh2)
                else:
                    pyautogui.scroll(-thresh2)
            #If four fingers are seen
            #Clicking on the screen
            if cnt==3:
                pyautogui.click()
            #If five fingers are seen
            #Right click on the screen
            if cnt==4:
                pyautogui.click(button='right')
            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print (cnt)
                    

        cv2.imshow('output', drawing)
    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('s'):  
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background has been Captured!!!')
    elif k == ord('r'):  
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')

        

