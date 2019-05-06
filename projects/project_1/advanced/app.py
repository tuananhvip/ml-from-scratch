import cv2
import numpy as np
from lenet import Lenet
import tensorflow as tf
import matplotlib.pyplot as plt

drawing = False # true if mouse is pressed
start_point = None
end_point = None

def transform_img(image, is_plotted=False):
    resized = tf.image.resize_images(image, (28, 28), tf.image.ResizeMethod.AREA)
    resized = resized.numpy().reshape((28, 28))
    if is_plotted:
        import matplotlib.pyplot as plt
        plt.imshow(resized, cmap='gray')
        plt.show()
    resized = resized / 255
    resized = resized.reshape((1, 28, 28, 1))
    return resized

lenet = Lenet(20, 64, tf.train.AdamOptimizer(learning_rate=0.001), tf.losses.softmax_cross_entropy)
lenet.load_model()

# mouse callback function
def draw_circle(event, x, y, _, __):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x,y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            cv2.line(img, start_point, end_point, 255, 15)
            start_point = end_point

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img_ = transform_img(img, True)
        pred = lenet.predict(img_)
        print(pred)

img = np.zeros((400, 400, 1), np.uint8)
cv2.namedWindow('Digit Recognition')
cv2.setMouseCallback('Digit Recognition', draw_circle)

while(1):
    cv2.imshow('Digit Recognition',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('m'):
        img = np.zeros((400, 400, 1), np.uint8)

cv2.destroyAllWindows()