# Interactive Wall Poster
# using camera and projector
# calibration + image registration + interaction

from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2 as cv
import time
import ctypes
import math
from skimage.util import random_noise

def main():
    print("main")
    
    global cap
    cap = cv.VideoCapture(0) # index of camera 0 or 1
    if cap is None or not cap.isOpened():
        print('Warning: unable to open camera')
    
    # USER INTERFACE
    user_interface()

    global H_projector
    global screen_w
    global screen_h
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [screen_w, screen_h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)] # screen size

    cv.namedWindow("result", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("result", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # CALIBRATION
    while True:
        H_projector = calibration() # homography
        # H_projector = np.load("H_projector.npy") # saved H for reuse if system stays static
        if H_projector is not None and len(H_projector) != 0: # if not received empty matrix
            break
        else:
            print("Did not find homography")
    
    _, blackframe = cap.read()
    blackframe.fill(0)
    white_bg = blackframe.copy()
    white_bg.fill(255)
    cv.imshow("result", white_bg) # always showing white background
    cv.waitKey(1)

    # IMAGE REGISTRATION
    global query_image
    query_image = cv.cvtColor(clone, cv.COLOR_BGR2GRAY) # query image (poster)
    global h1
    global w1
    h1, w1 = query_image.shape
    global T_poster
    T_poster, T_poster_inv = register_image() # transform
    
    cv.imshow("result", white_bg)
    cv.waitKey(1)
    time.sleep(3)
    _, frame = cap.read()
    prev_frame = getFrame(frame)
    h3, w3 = prev_frame.shape
    white_poster = prev_frame.copy()
    white_poster.fill(255) # for inserting later on a white background
    white_poster = np.stack((white_poster,)*3, axis=-1)

    # global change threshold
    img1 = prev_frame
    img0 = img1.copy()
    img0.fill(195)
    img0 = random_noise(img0, mode='s&p', amount=0.011)
    img0 = np.array(255 * img0, dtype=np.uint8)
    img0 = cv.blur(img0,(10,10))
    ncc_gl = get_ncc(img0, img1)
    ncc_gl = ncc_gl + 0.35 # a little more space in interval

    # MAIN LOOP FOR INTERACTIVITY
    print("main loop")
    while True:
        # check for START
        print("-----")
        timer = 0.806 # capture image with delay, change depending on sync
        time.sleep(timer)
        _, frame = cap.read()
        curr_frame = getFrame(frame)
        for region_point in range(0,len(refPt),2):
            # ncc(capture, white) is larger than ncc(capture, region) when we start interaction
            img0 = query_image[refPt[region_point][1]:refPt[region_point+1][1], refPt[region_point][0]:refPt[region_point+1][0]]
            img1 = img0.copy()
            img1.fill(195)
            img1 = random_noise(img1, mode='s&p', amount=0.011)
            img1 = np.array(255 * img1, dtype=np.uint8)
            img0 = cv.blur(img0,(10,10))
            img1 = cv.blur(img1,(10,10))
            curr_frame_region = curr_frame[refPt[region_point][1]:refPt[region_point+1][1], refPt[region_point][0]:refPt[region_point+1][0]]
            ncc_1 = get_ncc(curr_frame_region, img0)
            ncc_2 = get_ncc(curr_frame_region, img1)
            # a little more space in interval
            ncc_1 = ncc_1 - 0.15
            ncc_2 = ncc_2 + 0.15
            if ncc_2 < ncc_1:
                print(ncc_2, "<", ncc_1, "\n")
            if ncc_2 > ncc_1:
                print(ncc_2, ">", ncc_1, "\n")
                ncc_global = get_ncc(prev_frame, curr_frame)
                if ncc_global > ncc_gl:
                    print("ncc_global > ", ncc_gl, "\n")
                    # insert interactive content, read and transform video to play on region
                    region_video = cv.VideoCapture(videoNames[region_point])
                    if (region_video.isOpened()== False):
                        print("Error opening video file")
                    count_frame = 19 # for capturing image with delay
                    new_prev_frame = curr_frame
                    while(region_video.isOpened()):
                        vid_ret, vid_frame = region_video.read()
                        cv.normalize(vid_frame, vid_frame, 0, 255, cv.NORM_MINMAX)
                        if vid_ret == True:
                            count_frame = count_frame + 1
                            h4, w4, c4 = vid_frame.shape
                            pts1 = np.float32([[0, 0], [0, h4], [w4, h4], [w4, 0]]).reshape(-1, 1, 2) # video points
                            pts2 = np.float32([[refPt[region_point][0], refPt[region_point][1]], [refPt[region_point][0], refPt[region_point+1][1]], [refPt[region_point+1][0], refPt[region_point+1][1]], [refPt[region_point+1][0], refPt[region_point][1]]]).reshape(-1, 1, 2) # here i want the video to go
                            M = cv.getPerspectiveTransform(pts1, pts2)
                            dst = cv.warpPerspective(vid_frame,M,(w3, h3))
                            overlay = white_poster+dst
                            overlay_full = cv.warpPerspective(overlay, T_poster_inv, (w2, h2), borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))
                            cv.imshow("result", overlay_full)
                            if cv.waitKey(25) & 0xFF == ord('q'): #PRESS q to quit
                                break
                            # check for STOP
                            if count_frame % 20 == 0:
                                if count_frame == 20:
                                    time.sleep(1)
                                    _, frame = cap.read()
                                    new_prev_frame = getFrame(frame)
                                _, frame = cap.read()
                                new_curr_frame = getFrame(frame)
                                #ncc(capture, videoframe+region) is larger than ncc(capture, videoframe) when we stop transaction
                                img0 = query_image[refPt[region_point][1]:refPt[region_point+1][1], refPt[region_point][0]:refPt[region_point+1][0]]
                                h5, w5 = img0.shape
                                img1 = vid_frame.copy()
                                img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
                                img1 = cv.resize(img1, (w5, h5))
                                img2 = cv.addWeighted(img0,0.6,img1,0.4,-10)
                                img1 = cv.blur(img1,(10,10))
                                img2 = cv.blur(img2,(10,10))
                                new_curr_frame_region = new_curr_frame[refPt[region_point][1]:refPt[region_point+1][1], refPt[region_point][0]:refPt[region_point+1][0]]
                                ncc_1 = get_ncc(new_curr_frame_region, img1)
                                ncc_2 = get_ncc(new_curr_frame_region, img2)
                                # a little more space in interval
                                ncc_1 = ncc_1 - 0.02
                                ncc_2 = ncc_2 + 0.02
                                if ncc_2 < ncc_1:
                                    print(ncc_2, "<", ncc_1, "\n")
                                if ncc_2 > ncc_1:
                                    print(ncc_2, ">", ncc_1, ", break \n")
                                    break     
                                
                        else:
                            break
                    cv.imshow("result", white_bg)
                    cv.waitKey(1)
                    time.sleep(timer)
                    region_video.release()
                    _, frame = cap.read()
                    prev_frame = getFrame(frame)
                else:
                    print("ncc_global < ", ncc_gl, ", recalibration needed \n")
                    time.sleep(1) # a chance to remove the obstruction from frame
                    while True:
                        H_projector = calibration() #homography
                        if H_projector is not None and len(H_projector) != 0: # if not received empty matrix
                            break
                        else:
                            print("Did not find homography")
                    cv.imshow("result", white_bg)
                    cv.waitKey(1)
                    time.sleep(timer)
                    _, frame = cap.read()
                    prev_frame = getFrame(frame)
                    T_poster, T_poster_inv = register_image()
        if cv.waitKey(25) & 0xFF == ord('q'): #PRESS q to quit
            break

    cap.release()
    cv.destroyAllWindows()

# USER INTERFACE
# choice of poster, active regions, interactive content per each region
def user_interface():
    print("user interface")

    Tk().withdraw()
    postername = askopenfilename()

    global refPt
    global image
    global videoNames
    global clone
    refPt = [] # list of reference points
    image = cv.imread(postername)
    videoNames = [] # list of video names
    clone = image.copy()
    cv.namedWindow("user interface", cv.WINDOW_NORMAL)    
    cv.setMouseCallback("user interface", click_and_crop) # mouse callback function
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv.imshow("user interface", image)
        key = cv.waitKey(1) & 0xFF
        if key == ord("r"): #PRESS r to reset the cropping region
            image = clone.copy()
        elif key == ord("c"): #PRESS c to break from the loop
            break
    # close all open windows
    cv.destroyAllWindows()

def click_and_crop(event, x, y, flags, param):
    # if the left mouse button was clicked, record the starting (x,y) coordinates
    if event == cv.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
    # check if the left mouse button was released
    elif event == cv.EVENT_LBUTTONUP:
        # record the  ending (x,y) coordinates
        refPt.append((x, y))
        # draw a rectangle around the region of interest
        cv.rectangle(image, refPt[len(refPt)-2], refPt[len(refPt)-1], (0, 255, 0), 2)
        cv.imshow("user interface", image)
        # choose interactive content to match the chosen region
        Tk().withdraw()
        videoname = askopenfilename()
        videoNames.append(videoname)
        videoNames.append("") #one empty so it matches the two points from refPt

# CALIBRATION
# projector calibration by projecting and detecting circles
def calibration():
    print("calibration")
    timer = 1 # capture image with delay
    img = np.zeros((screen_h,screen_w), np.uint8)
    cv.imshow("result", img)
    cv.waitKey(1)
    time.sleep(timer)
    _, frame = cap.read()
    scene1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #reference = no circle being shown

    points_og = [] # points drawn
    points_detected = [] # points detected
    radius = 100
    x = 0 + radius
    y = 0 + radius
    jumpX = int((screen_w - 2*radius)/3)
    jumpY = int((screen_h - 2*radius)/3)
    for i in range(4):
        for j in range(4):
            img.fill(0)
            cv.circle(img, (x, y), radius, (255, 255, 255), -1)
            cv.imshow("result", img)
            cv.waitKey(1)
            time.sleep(timer)
            _, frame = cap.read()
            if frame is not None:
                scene2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                diff = cv.absdiff(scene1, scene2)
                diff[diff > 30] = 255
                ret,gray = cv.threshold(diff,127,255,0)

                contours,hierarchy = cv.findContours(gray,2,1)
                cnt = contours
                if len(cnt) == 0:
                    x = x + jumpX
                    break
                big_contour = []
                max = 0
                for i in cnt:
                    area = cv.contourArea(i) # find the contour with biggest area
                    if (area > max):
                        max = area
                        big_contour = i
                if max > 1000: # ignore small diffs, contours
                    ((xx, yy), r) = cv.minEnclosingCircle(big_contour)
                    points_og.append((x, y))
                    points_detected.append((xx, yy))
                x = x + jumpX
            
        x = 0 + radius
        y = y + jumpY
    points_og = np.array(points_og)
    points_detected = np.array(points_detected)
    H, _ = cv.findHomography(points_detected, points_og, cv.RANSAC, 5.0)
    frame.fill(255)
    cv.imshow("result", frame)
    cv.waitKey(1)
    np.save("H_projector.npy", H)
    return H

# IMAGE REGISTRATION
# find poster in image capture
def register_image():
    print("register image")
    _, frame = cap.read()
    frame.fill(255)
    cv.imshow("result", frame)
    cv.waitKey(1)
    sift = cv.xfeatures2d.SIFT_create()
    kp_image, desc_image = sift.detectAndCompute(query_image, None) # query image keypoints
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv.FlannBasedMatcher(index_params, search_params)
    global h2
    global w2
    while True:
        _, frame = cap.read()
        grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        train_image = cv.warpPerspective(grayframe, H_projector, (screen_w, screen_h)) # train image keypoints
        h2, w2 = train_image.shape
        kp_grayframe, desc_grayframe = sift.detectAndCompute(train_image, None)
        if desc_grayframe is not None and kp_grayframe is not None:
            try:
                matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
            except cv.error:
                print("cv2.error: no keypoints found in video capture.")
            if matches is not None:
                good = []
                for m,n in matches:
                    if m.distance < 0.8*n.distance: #ratio test
                        good.append(m)
                if len(good) > 15:
                    try:
                        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    except IndexError:
                        print("IndexError: train_pts...")
                    matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
                    matches_mask = mask.ravel().tolist()
                    pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
                    if mask is not None and matrix is not None:
                        dst = cv.perspectiveTransform(pts, matrix)
                        homography = cv.polylines(frame, [np.int32(dst)], True, (255, 255, 255), 3)
                        # focus only on poster
                        aligned = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
                        M = cv.getPerspectiveTransform(dst, aligned)
                        M_inv = cv.getPerspectiveTransform(aligned, dst)
                        return M, M_inv
                    else:
                        print("no mask or matrix found")
                else:
                    print("len(good) was <=15")
            else:
                print("no matches found")
        else:
            print("no descriptors found")

# we warp image capture by applying H_projector and T_poster to focus on poster
def getFrame(img):
    projected = cv.warpPerspective(img, H_projector, (screen_w, screen_h)) # h2, w2 are the shape
    poster = cv.warpPerspective(projected, T_poster, (w1, h1)) # this is now only watching the poster
    frame = cv.cvtColor(poster, cv.COLOR_BGR2GRAY)
    return frame

# get ncc coefficient
def get_ncc(mat1, mat2):
    ncc = 0.0
    stds = np.std(mat1)*np.std(mat2)
    if stds != 0:
        ncc = np.mean(np.multiply((mat1-np.mean(mat1)),(mat2-np.mean(mat2))))/stds
    return ncc 

# Application entry point -> call main()
main()
