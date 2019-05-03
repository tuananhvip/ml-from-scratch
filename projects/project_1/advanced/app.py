import cv2
import numpy as np
from lenet import Lenet
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

drawing = False # true if mouse is pressed
ix, iy = -1, -1

def transform_img(image):
    resized = tf.image.resize_images(image, (28, 28), tf.image.ResizeMethod.AREA)
    resized = resized.numpy().reshape((28, 28))
    plt.imshow(resized, cmap='gray')
    plt.show()
    scaler = StandardScaler()
    scaler.fit(resized)
    resized = scaler.transform(resized)
    resized = resized.reshape((1, 28, 28, 1))
    return resized

lenet = Lenet(20, 64, tf.train.AdamOptimizer(learning_rate=0.001), tf.losses.softmax_cross_entropy)
lenet.load_model()

# mouse callback function
def draw_circle(event, x, y, _, __):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            cv2.circle(img, (x, y), 5, 255, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 5, 255, -1)
        img_ = transform_img(img)
        print(img_)
        pred = lenet.predict(img_)
        print(pred)

img = np.zeros((512, 512, 1), np.uint8)
cv2.namedWindow('Digit Recognition')
cv2.setMouseCallback('Digit Recognition', draw_circle)

while(1):
    cv2.imshow('Digit Recognition',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('m'):
        img = np.zeros((512, 512, 1), np.uint8)

cv2.destroyAllWindows()