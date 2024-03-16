import json
import cv2
import numpy as np

with open("front/run_01.json") as f:
    data = json.load(f)

points = data[0]["people"][0]["pose_keypoints_2d"]
print(len(points))

height = data[0]["canvas_height"]
width  = data[0]["canvas_width"]

img = np.zeros((height,width,3))

skip = 3
for i in range(len(points) // skip):
    perc = i / (len(points)//skip)
    if points[i*skip+2] < 0.5:
        print(f"found missing point, skipping {i}")
    loc = (int(points[i*skip]*width), int(points[i*skip+1]*height))
    img = cv2.circle(img, loc, 5, (255*perc,255*(1-perc),255))

cv2.imwrite("points.png", img)
