import cv2
import numpy as np
import matplotlib.pyplot as plt


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    rows, columns= im.shape
    newImage= np.zeros((rows, columns), dtype=np.uint8)
    #we create a zero padded matrix which will be of size rows + 2*radius, columns + 2*radius
    zeroPadded = np.zeros((rows + 2*radius, columns + 2*radius), dtype=np.float64)
    zeroPadded[radius:-radius,radius:-radius]=im

    for i in range(rows):
        #create the windows
        window = np.zeros((1 + 2 * radius, 1 + 2 * radius), dtype=np.float64)
        for j in range(columns):
            #initialize the windows
            window = zeroPadded[i:i+2*radius+1, j:j+2*radius+1]
            #create variables to save the sums needed for the formula to calculate value of pixel
            gigsSum= np.float64(0)
            gigsWinSum = np.float64(0)
            constant= window[radius,radius]
            gi= np.exp(-((window-constant)**2)/(2*(stdIntensity**2)))
            #create matrix to calculate Gs.
            x, y= np.meshgrid(np.arange(2*radius+1), np.arange(2*radius+1))
            gs=np.exp(-((x - radius)**2 + (y - radius)**2)/(2*(stdSpatial)**2))
            #calculate the sums.
            gigsSum = np.sum(np.multiply(gi, gs))
            gigsWinSum=np.sum(np.multiply(np.multiply(gi, gs), window))
            #determine each pixel value
            pixelValue= gigsWinSum/gigsSum
            #fill the new image pixel.
            newImage[i,j]=pixelValue.astype(np.uint8)
    return newImage



# change this to the name of the image you'll try to clean up
original_image_path = r'C:\Users\majd_\OneDrive\Desktop\CS\IP\HW3\q2\balls.jpg'
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
clear_image_b = clean_Gaussian_noise_bilateral(image, 3, 6, 30)
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.imshow(clear_image_b, cmap='gray')
plt.show()
#cv2.imwrite(r"C:\Users\majd_\OneDrive\Desktop\CS\IP\HW3\q2\ballsfixed.jpg", clear_image_b)
print(image[100,100],clear_image_b[100,100])
