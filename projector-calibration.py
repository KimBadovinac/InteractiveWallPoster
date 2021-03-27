# Projector calibration by projecting and detecting circles

import numpy as np
import cv2 as cv
import time
import ctypes

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)] # screen size

timer = 1 # capture image with delay
img = np.zeros((h,w), np.uint8)

cap = cv.VideoCapture(0) # index of camera 0 or 1
if cap is None or not cap.isOpened():
    print('Warning: unable to open camera')

cv.namedWindow("result", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("result", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

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
jumpX = int((w - 2*radius)/3)
jumpY = int((h - 2*radius)/3)
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

# warp camera capture with computed homography
# check if projected content matches real content
_, frame = cap.read()
result = cv.warpPerspective(frame, H, (w, h))
cv.imshow("result", result)
cv.waitKey(-1)

cv.destroyAllWindows()
cap.release()
#np.save("H_projector.npy", H)
