
# coding: utf-8

# In[1]:

import numpy as np
import cv2
from sklearn.utils import shuffle

def load_data(file):
    return np.load(file)

def load_samples(file):
    data = load_data(file)
    IMAGES = data[0]
    LABELS = data[1]
    IMAGES, LABELS = shuffle(np.concatenate((IMAGES[:455], IMAGES[755:2155])), 
                             np.concatenate((LABELS[:455], LABELS[755:2155])))
    #IMAGES, LABELS = shuffle(IMAGES[:2155], LABELS[:2155])
    #IMAGES, LABELS = shuffle(IMAGES, LABELS)
    return IMAGES, LABELS


def norm(val, dev):
    half = dev / 2
    return (val - half) / half

def prepare_label(label):
    c,x1,y1,x2,y2 = label
    if c != 0:
        x1, x2 = norm(x1, 48), norm(x2, 48)
        y1, y2 = norm(y1, 64), norm(y2, 64)
    return (c, x1, y1, x2, y2)

def normalize_labels(labels):
    for i in range(len(labels)):
        labels[i] = prepare_label(labels[i])

def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def gray_norm(frame):
    return cv2.equalizeHist(gray(frame))

def hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def normalize_picture(image):
    #image[:,:,0] = gray(image)
    #image[:,:,1] = gray_norm(image)
    #image[:,:,2] = hsv(image)[:,:,0] # h channel
    hsvi = hsv(image)
    hsvi[:,:,2] = cv2.equalizeHist(hsvi[:,:,2]) # h channel
    #image = hsv(image)# h channel
    image = hsvi
    image = image.astype(np.float32) * 1.0/255.
    return image

def normalize_pictures(images):
    for i in range(len(images)):
        images[i] = normalize_picture(images[i])
        
def load_and_normilize(file):
    IMAGES, LABELS = load_samples(file)
    normalize_pictures(IMAGES)
    normalize_labels(LABELS)
    # reshape arrays for tensorflow
    IMAGES = np.concatenate(IMAGES).reshape(IMAGES.shape[0], *IMAGES[0].shape)
    LABELS = np.concatenate(LABELS).reshape(LABELS.shape[0], len(LABELS[0]))
    X_train = IMAGES[:-500]
    Y_train = LABELS[:-500]
    X_test = IMAGES[-500:]
    Y_test = LABELS[-500:]
    return (X_train, Y_train, X_test, Y_test)
