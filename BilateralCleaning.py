# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
	# Your code goes here
    rows, columns= im.shape
    newImage= np.zeros((rows, columns), dtype=np.uint8)
    zeroPadded = np.zeros((rows + 2*radius, columns + 2*radius), dtype=np.float64)
    zeroPadded[radius:-radius,radius:-radius]=im
    #print(zeroPadded[rows,columns])
    #print(zeroPadded.shape)
    for i in range(rows):
        window = np.zeros((1 + 2 * radius, 1 + 2 * radius), dtype=np.float64)

        for j in range(columns):

            window = zeroPadded[i:i+2*radius+1, j:j+2*radius+1]
            #print(window.shape)
            #print(window)
            gigsSum= np.float64(0)
            gigsWinSum = np.float64(0)
            constant= window[radius,radius]
            gi= np.exp(-((window-constant)**2)/(2*(stdIntensity**2)))
            x, y= np.meshgrid(np.arange(2*radius+1), np.arange(2*radius+1))
            gs=np.exp(-((x - radius)**2 + (y - radius)**2)/(2*(stdSpatial)**2))
            #print(gi.shape, gs.shape)
            gigsSum = np.sum(np.multiply(gi, gs))
            gigsWinSum=np.sum(np.multiply(np.multiply(gi, gs), window))

            pixelValue= gigsWinSum/gigsSum
            #print(pixelValue)
            newImage[i,j]=pixelValue.astype(np.uint8)
    return newImage




# change this to the name of the image you'll try to clean up
original_image_path = r'C:\Users\majd_\OneDrive\Desktop\CS\IP\HW3\q2\taj.jpg'
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

clear_image_b = clean_Gaussian_noise_bilateral(image, 9, 20, 20)
#print(clear_image_b.shape)
plt.subplot(121)
plt.imshow(image, cmap='gray')
print(image.dtype)
plt.subplot(122)
plt.imshow(clear_image_b, cmap='gray')

plt.show()
print(image[100,100],clear_image_b[100,100])