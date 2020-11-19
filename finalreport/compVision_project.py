import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 0
high_threshold = 300
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
cv2.imshow('imageCanny',edges)
cv2.waitKey(0)



rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30  # minimum number of pixels making up a line
max_line_gap = 30  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# Draw the lines on the  image
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

cv2.imshow('imageHough',lines_edges)
cv2.waitKey(0)




'''
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
sift = cv2.SIFT_create()

kp1 = sift.detect(gray,None)
imgSift=cv2.drawKeypoints(gray,kp1,None)
cv2.imshow('imageSift',imgSift)
cv2.waitKey(0)

# find and draw the keypoints
kp2 = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp2, None, color=(255,0,0))

# Print all default params
print ("Threshold: ", fast.getThreshold())
print ("nonmaxSuppression: ", fast.getNonmaxSuppression())
print ("neighborhood: ", fast.getType())
print ("Total Keypoints with nonmaxSuppression: ", len(kp2))

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
#fast.setNonmaxSuppression(0)
#kp = fast.detect(img,None)

print ("Total Keypoints without nonmaxSuppression: ", len(kp2))

img3 = cv2.drawKeypoints(img, kp2, None, color=(255,0,0))

#cv2.imwrite('fast_false.png',img3)

cv2.imshow('imageFast',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''