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

for i in range(len(points) // 2):
    perc = i / (len(points)//2)
    loc = (int(points[i*2]*width), int(points[i*2+1]*height))
    img = cv2.circle(img, loc, 5, (255*perc,255*(1-perc),255))

cv2.imwrite("points.png", img)
