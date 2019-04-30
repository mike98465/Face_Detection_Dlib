import cv2
import dlib
from PIL import Image
import sys
import numpy as np

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

f = 'test.jpg'
image = cv2.imread('test.jpg',cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Processing file: {}".format(f))
img = dlib.load_rgb_image(f)
# The 1 in the second argument indicates that we should upsample the image
# 1 time.  This will make everything bigger and allow us to detect more
# faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
  print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
  i, d.left(), d.top(), d.right(), d.bottom()))
  cv2.rectangle(image, (d.left(),d.top()), (d.right(),d.bottom()), (0, 255, 0), 5)

win.clear_overlay()
win.set_image(img)
win.add_overlay(dets)
dlib.hit_enter_to_continue()


# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.
if (len(sys.argv[1:]) > 0):
    img = dlib.load_rgb_image(f)
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
           d, scores[i], idx[i]))