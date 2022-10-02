#traffic signs detection, advanced lane detection, pothole detection


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset
from moviepy.editor import VideoFileClip

import pygame
import cv2 as cv
import time
import smtplib

import pandas as pd

import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
#from sklearn import train_test_split
from keras.utils import to_categorical
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout


# traffic signs detection
data=[]
labels=[]
classes=43
cur_path=os.getcwd()

for i in range(classes):
    path=os.path.join(cur_path,'train',str(i))
    images=os.listdir(path)

    for a in images:
        try:
            image=Image.open(path +'\\'+ a)
            image=image.resize((30,30))
            image=np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")
data=np.array(data)
labels=np.array(labels)
print(data.shape,labels.shape)
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
y_train=to_categorical(y_train,43)
y_test=to_categorical(y_test,43)
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu',input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
epochs=3
history=model.fit(X_train,y_train,batch_size=64,epochs=epochs,validation_data=(X_test,y_test))
plt.figure(0)
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.figure()
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
from sklearn.metrics import accuracy_score
import pandas as pd
y_test=pd.read_csv('Test.csv')
labels=y_test["ClassId"].values
imgs=y_test["Path"].values
data=[]
for img in imgs:
    image=Image.open(img)
    image=image.resize((30,30))
    data.append(np.array(image))
X_test=np.array(data)
pred=(model.predict(X_test) > 0.5).astype("int32")
from sklearn.metrics import accuracy_score
accuracy_score(labels,pred)
model.save('traffic_classifier.h5')


#----------------

#lane detectiion
# Global variables (just to make the moviepy video annotation work)
with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']
window_size = 5  # how many frames for line smoothing
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # did the fast line fit detect the lines?
left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
left_lane_inds, right_lane_inds = None, None  # for calculating curvature


# MoviePy video annotation will call this function
def annotate_image(img_in):
	"""
	Annotate the input image with lane line markings
	Returns annotated image
	"""
	global mtx, dist, left_line, right_line, detected
	global left_curve, right_curve, left_lane_inds, right_lane_inds

	# Undistort, threshold, perspective transform
	undist = cv2.undistort(img_in, mtx, dist, None, mtx)
	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
	binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)

	# Perform polynomial fit
	if not detected:
		# Slow line fit
		ret = line_fit(binary_warped)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Get moving average of line fit coefficients
		left_fit = left_line.add_fit(left_fit)
		right_fit = right_line.add_fit(right_fit)

		# Calculate curvature
		left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

		detected = True  # slow line fit always detects the line

	else:  # implies detected == True
		# Fast line fit
		left_fit = left_line.get_fit()
		right_fit = right_line.get_fit()
		ret = tune_fit(binary_warped, left_fit, right_fit)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Only make updates if we detected lines in current frame
		if ret is not None:
			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']

			left_fit = left_line.add_fit(left_fit)
			right_fit = right_line.add_fit(right_fit)
			left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
		else:
			detected = False

	vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)

	# Perform final visualization on top of original undistorted image
	result = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

	return result


def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)


#----
#pothole detection

im = cv2.imread('index4.jpg')
# CODE TO CONVERT TO GRAYSCALE


gray1 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# save the image
cv2.imwrite('graypothholeresult.jpg', gray1)
plt.subplot(331),plt.imshow(gray1, cmap='gray'),plt.title('GRAY')
plt.xticks([]), plt.yticks([])
#CONTOUR DETECTION CODE
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)

_, contours1, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
_, contours2, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#img1 = im.copy()
img2 = im.copy()

#out = cv2.drawContours(img1, contours1, -1, (255,0,0), 2)
out = cv2.drawContours(img2, contours2, -1, (250,250,250),1)
#out = np.hstack([img1, img2])


img = cv2.imread('index2.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
_, contours,hierarchy = cv2.findContours(thresh, 1, 2) 
cnt = contours[0]
M = cv2.moments(cnt)

#print M
perimeter = cv2.arcLength(cnt,True)
#print perimeter
area = cv2.contourArea(cnt)
#print area
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
#print epsilon
#print approx
for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 100 or rect[3] < 100: continue
    print(cv2.contourArea(c))
    x,y,w,h = rect
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),8)
    cv2.putText(img2,'Moth Detected',(x+w+40,y+h),0,2.0,(0,255,0))

#cv2.waitKey()  
#cv2.destroyAllWindows()
k = cv2.isContourConvex(cnt)

#to check convexity
print(k)
#blur
blur = cv2.blur(gray1,(5,5))
#guassian blur 
gblur = cv2.GaussianBlur(gray1,(5,5),0)
#median 
median = cv2.medianBlur(gray1,5)
#erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(median,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 5)
#erosion followed dilation
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
#canny edge detection
edges = cv2.Canny(dilation,9,220)  
#plotting using matplotlib
plt.subplot(332),plt.imshow(blur,cmap = 'gray'),plt.title('BLURRED')
plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(gblur,cmap = 'gray'),plt.title('guassianblur')
plt.xticks([]), plt.yticks([])        
plt.subplot(334),plt.imshow(median,cmap = 'gray'),plt.title('Medianblur')
plt.xticks([]), plt.yticks([])
plt.subplot(337),plt.imshow(dilation,cmap = 'gray')
plt.title('dilated Image'), plt.xticks([]), plt.yticks([])
plt.subplot(338),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(erosion,cmap = 'gray'),plt.title('EROSION')
plt.xticks([]), plt.yticks([])
plt.subplot(336),plt.imshow(closing,cmap = 'gray'),plt.title('closing')
plt.xticks([]), plt.yticks([])
plt.show()






if __name__ == '__main__':
	# Annotate the video
	annotate_video('test_video.mp4', 'tout.mp4')

	# Show example annotated image on screen for sanity check
	img_file = 'test_images/test2.jpg'
	img = mpimg.imread(img_file)
	result = annotate_image(img)
	result = annotate_image(img)
	result = annotate_image(img)
	plt.imshow(result)
	plt.show()
