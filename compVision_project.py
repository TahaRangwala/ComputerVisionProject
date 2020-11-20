import numpy as np
import cv2
from matplotlib import pyplot as plt
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

params = {}
params["wide"] = 20
params["image"] = "Sample Input Files/sample3.jpg"
img = cv2.imread(params["image"])
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# This section defines the constants and thresholds used throughout the code.
contour_threshold = 20
acceptance_thresh = 80
rejection_thresh = 300
edges = cv2.Canny(blur_gray, acceptance_thresh, rejection_thresh)
cv2.imwrite('imageCannyOut.png',edges)
cv2.waitKey(0)

# These are parameters we tuned for the hough transform so that the edges of the walls would
# be found in most sample images.
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 30
max_line_gap = 30
image_lines = np.copy(img) * 0

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

# this draws the hough lines on the image.
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(image_lines,(x1,y1),(x2,y2),(255,0,0),5)

# this function uses the edges that we detected to find a list of contours.
# This list is then used with to generate a pixel scalar.
listContours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE)
# this is a list of contours obtained
listContours = imutils.grab_contours(listContours)
(listContours, _) = contours.sort_contours(listContours)
ppm = None

counter = 0
for c in listContours:
    # This checks to ensure that the shape is above a pre-defined area of 85 "inches".
    if cv2.contourArea(c) < contour_threshold:
        continue
    orig = img.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    for xCoord, yCoord in box:
        cv2.circle(orig, (int(xCoord), int(yCoord)), 5, (0, 0, 255), -1)

    # This code calculates the midpoints between the two detected edge pairs (horizontal pair or vertical pair)
    (topLeft, topRight, bottomRight, bottomLeft) = box
    (midX1, midY1) = midpoint(topLeft, topRight)
    (midX2, midY2) = midpoint(bottomLeft, bottomRight)
    (midX3, midY3) = midpoint(topLeft, bottomLeft)
    (midX4, midY4) = midpoint(topRight, bottomRight)

    # This code draws the midpoints
    cv2.line(orig, (int(midX1), int(midY1)), (int(midX2), int(midY2)),
        (200, 0, 120), 2)
    cv2.line(orig, (int(midX3), int(midY3)), (int(midX4), int(midY4)),
        (200, 0, 120), 2)

    # this code finds the ditances between the midpoints using the euclidean distance calculation.
    firstDistance = dist.euclidean((midX1, midY1), (midX2, midY2))
    secondDistance = dist.euclidean((midX3, midY3), (midX4, midY4))
    
    # ppm is a measure of pixel scaling:
    if ppm == None:
        ppm = secondDistance / params["wide"]
        
    # these variables are the scaled
    firstDimension = firstDistance / ppm
    secondDimension = secondDistance / ppm
    cv2.putText(orig, str(firstDimension) + ' inches',
        (int(midX1 - 15), int(midY1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 0), 2)
    cv2.putText(orig, str(secondDimension) + ' inches',
        (int(midX4 + 10), int(midY4)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 0), 2)
    # This adds the hough edges to each output image
    egde_lines = cv2.addWeighted(img, 0.8, image_lines, 1, 0)
    # This adds the midpoint edges to each output image
    orig = cv2.addWeighted(orig, 0.8, image_lines, 1, 0)
    # writes the image to an output file
    cv2.imwrite("ImageOut%i.png" % counter, orig)
    counter += 1
    cv2.waitKey(0)

# This adds the hough lines to the original image
egde_lines = cv2.addWeighted(img, 0.8, image_lines, 1, 0)
# saves the image without the measurements
cv2.imwrite('imageHough.png', egde_lines)
cv2.waitKey(0)
