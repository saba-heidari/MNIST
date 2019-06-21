# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:21:09 2019

@author: Saba
"""

from sklearn import datasets
from sklearn.svm import LinearSVC
import cv2
from skimage import feature
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX

hog_list =[]


mnist_dataset = datasets.fetch_openml("mnist_784")
features = np.array(mnist_dataset.data, "uint8")
labels = np.array(mnist_dataset.target, "int")

# create list of features for all images:
for image in features:
    hog = feature.hog(image.reshape(28, 28), visualise = False)
    hog_list.append(hog)
    
  # Classifier
    
clf = LinearSVC()

clf.fit(hog_list, labels)


""" Pre Process"""


img = cv2.imread("digit1.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY_INV)

_ , cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    M =cv2.moments(c)
    if M["m00"] > 20:
        (x, y, w, h) = cv2.boundingRect(c)
        crop_img = img [y:y+h, x:x+w]
        resize = cv2.resize(crop_img, (28, 28))
        feature_hog = feature.hog(resize) 
        predicted_label = clf.predict(np.array([feature_hog]))
        cv2.putText(img,"{}".format(int(predicted_label)), (x-10, y-10), font, 1.5, (0, 255, 0), 3 )
        
        cv2.rectangle(img, (x, y), (x+w , y+h), (0, 255, 0), 3)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
