#Experimental Script for Sobel Edge Dection and Disparity

imgR = cv2.imread('dispR.jpg', 0)
imgL = cv2.imread('dispL.jpg', 0)

#imgR = cv2.GaussianBlur(imgR,(kernel_size, kernel_size),0)
#imgL = cv2.GaussianBlur(imgL,(kernel_size, kernel_size),0)

# creates StereoBm object  
stereo = cv2.StereoBM_create(numDisparities = 128,
                            blockSize = 5)
 
# computes disparity
disparity = stereo.compute(imgL, imgR)
 
# displays image as grayscale and plotted
plt.imshow(disparity, 'gray')
plt.show()

#############   sobel   #####################
img = cv2.imread('room2.jpg',0)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=31)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=31)
sobel = sobelx+sobely

plt.imshow(sobel, 'gray')
plt.show()