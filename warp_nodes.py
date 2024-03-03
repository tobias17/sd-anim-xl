from torch import Tensor
import numpy as np
import cv2
import math
from typing import Tuple

hue_index = {
    'head': 0,
    'leg_left_lower': 20,
    'leg_left_upper': 40,
    'torso_left': 60,
    'leg_right_lower': 80,
    'leg_right_upper': 100,
    'torso_right': 120,
    'arm_left_lower': 140,
    'arm_left_upper': 160,
    'arm_right_lower': 180,
    'arm_right_upper': 200,
    'shoulder_left': 220,
    'shoulder_right': 240,
}

def to_circle(angle, normalize_to=0) -> float:
    while angle >  math.pi + normalize_to:  angle -= 2*math.pi
    while angle < -math.pi + normalize_to:  angle += 2*math.pi
    return angle

class Vector2:
    def __init__(self, point:Tuple[float,float]):
        self.x = point[0]
        self.y = point[1]
    def point(self) -> Tuple[float,float]: return (self.x, self.y)
    def point_int(self) -> Tuple[int,int]: return (int(self.x), int(self.y))
    def __sub__(self, o) -> 'Vector2': return Vector2([self.x - o.x, self.y - o.y])
    def __add__(self, o) -> 'Vector2': return Vector2([self.x + o.x, self.y + o.y])
    def angle(self) -> float: return math.atan2(self.y, self.x)
    def length(self) -> float: return math.sqrt(self.x**2 + self.y**2)
    def __str__(self) -> str: return f"Point(x={self.x},y={self.y})"

    def __mul__(self, o) -> 'Vector2':
        if isinstance(o, type(self)): return Vector2([self.x * o.x, self.y * o.y])
        else:                         return Vector2([self.x * o,   self.y * o  ])

    def normalize(self, length=1) -> 'Vector2':
        assert length != 0, "Can not normalize a vector to length 0!"
        assert self.x != 0 or self.y != 0, "Can not normalize a vector of length 0!"
        div = self.length() / float(length)
        return Vector2([self.x / div, self.y / div])

class TransformComponent:
    transform_offset = 40
    tp_1, tp_2 = None, None

    def __init__(self, p1:Vector2, p2:Vector2):
        self.p1 = p1
        self.p2 = p2
        self.vec = p2 - p1
    
    def draw_boundary_on(self, draw_img, ref_img, angle):
        root = self.p1
        start_angle = self.vec.angle()*180/math.pi + (90 if angle > 0 else 270)
        end_angle = start_angle + angle
        move_vec = Vector2([math.cos(start_angle*math.pi/180), math.sin(start_angle*math.pi/180)])
        for i in range(1, 50+1):
            point = (root + (move_vec * i)).point_int()
            color = [int(v) for v in list(ref_img[point[1], point[0]])]
            cv2.ellipse(draw_img, root.point_int(), (i,i), 0, start_angle, end_angle, color, 2)

    def get_transform_points(self):
        points = []
        for is_left in [True, False]:
            for is_first in [True, False]:
                dir_vec = (self.p1 - self.p2 if is_first else self.p2 - self.p1).normalize(self.transform_offset)
                dir_vec = Vector2( [(1 if is_left else -1) * dir_vec.y, (-1 if is_left else 1) * dir_vec.x] )
                points.append(((self.p1 if is_first else self.p2) + dir_vec).point())
        return np.float32(points)

    def transform_image(self, img, comp, size):
        self.tp_1, comp.tp_2 = self.get_transform_points(), comp.get_transform_points()
        M = cv2.getPerspectiveTransform(self.tp_1, comp.tp_2)
        return cv2.warpPerspective(img, M, size)

    def draw_debug_on(self, img, color):
        cv2.line(img, self.p1.point_int(), (self.p1 + self.vec).point_int(), color, 4)

class TransformBoundary:
    dist = 1024

    def __init__(self, comp1:TransformComponent, comp2:TransformComponent):
        self.comp1 = comp1
        self.comp2 = comp2

        self.diff = to_circle(comp1.vec.angle() - comp2.vec.angle()) / 2
        self.angle = math.pi/2 + comp1.vec.angle() - self.diff
        self.vec = Vector2([math.cos(self.angle), math.sin(self.angle)])
    
    def draw_mask_on(self, img, color):
        angle = self.angle
        mod_dist = Vector2([math.cos(angle - math.pi/2) * self.dist, math.sin(angle - math.pi/2) * self.dist])
        cv2.line(img, (self.comp1.p2 + mod_dist - self.vec*self.dist).point_int(), (self.comp1.p2 + mod_dist + self.vec*self.dist).point_int(), color, int(self.dist*2))

    def draw_debug_on(self, img, p1, length, color, size):
        cv2.line(img, (p1 + (self.vec * length)).point_int(), (p1 - (self.vec * length)).point_int(), color, size)
        cv2.circle(img, (p1 + (self.vec * length)).point_int(), 5, (0, 0, 255, 255), -1)

class OpenPoseWarp:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "stretch_image":  ("IMAGE",),
                "stretch_pose":   ("IMAGE",),
                "body_mask":      ("MASK",),
                "right_arm_mask": ("MASK",),
                "left_arm_mask":  ("MASK",),
                "right_leg_mask": ("MASK",),
                "left_leg_mask":  ("MASK",),
                "target_pose":    ("IMAGE",),
            }
        }
    
    CATEGORY = "PoseWarp"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("target_pose",)
    FUNCTION = "open_pose_warp"

    def scale_point(self, x, y) -> Tuple[int,int]:
        return int(x*self.rescale[1]), int(y*self.rescale[0])

    def extract_position(self, hsv_pose_img:np.ndarray, body_part:str) -> Tuple[Tuple[int,int],Tuple[int,int]]:
        hue = hue_index.get(body_part, None)
        assert hue is not None, f"failed to find body part key '{body_part}' in hue_index dict"
        mask = cv2.inRange(hsv_pose_img, np.array([hue-2,0.98,0.58]), np.array([hue+2,1.02,0.62]))
        where = np.where(mask > 200)

        x1_h, x2_h = min(where[1]), max(where[1])
        y1_h, y2_h = min(np.where(mask[:,x1_h:x1_h+2] > 200)[0]), max(np.where(mask[:,x2_h-1:x2_h+1] > 200)[0])

        y1_v, y2_v = min(where[0]), max(where[0])
        x1_v, x2_v = min(np.where(mask[y1_v:y1_v+2,:] > 200)[1]), max(np.where(mask[y2_v-1:y2_v+1,:] > 200)[1])

        if ((x2_h-x1_h)**2 + (y2_h-y1_h)**2) > ((x2_v-x1_v)**2 + (y2_v-y1_v)**2):
            return self.scale_point(x1_h, y1_h), self.scale_point(x2_h, y2_h)
        return self.scale_point(x1_v, y1_v), self.scale_point(x2_v, y2_v)

    def clean_mask(self, mask:Tensor) -> np.ndarray:
        assert mask.shape[0], f"masks must be of batch size 1, found {mask.shape[0]}"
        mask_np = mask[0].numpy()
        return mask_np.reshape((*mask_np.shape,1))

    def open_pose_warp(self, stretch_image:Tensor, stretch_pose:Tensor, body_mask:Tensor, right_arm_mask:Tensor, left_arm_mask:Tensor, right_leg_mask:Tensor, left_leg_mask:Tensor, target_pose:Tensor):
        assert stretch_image.shape[0] == 1 and stretch_pose.shape[0] == 1, "cannot have a batch larger than 1 for stretch image and pose"
        stretch_image = stretch_image[0].numpy()
        stretch_pose  = stretch_pose[0].numpy()
        if stretch_pose.shape != stretch_image.shape:
            self.rescale = list(stretch_image.shape[i] / stretch_pose.shape[i] for i in range(2))
        else:
            self.rescale = (1.0,1.0)

        body_mask      = self.clean_mask(body_mask)
        right_arm_mask = self.clean_mask(right_arm_mask)
        left_arm_mask  = self.clean_mask(left_arm_mask)
        right_leg_mask = self.clean_mask(right_leg_mask)
        left_leg_mask  = self.clean_mask(left_leg_mask)

        hsv_pose_img = cv2.cvtColor(stretch_pose, cv2.COLOR_BGR2HSV)

        p1_u, p2_u = self.extract_position(hsv_pose_img, 'leg_right_upper')
        p1_l, p2_l = self.extract_position(hsv_pose_img, 'leg_right_lower')

        background_img = np.ones(stretch_image.shape) * 0.5

        left_leg_img = stretch_image*left_leg_mask + background_img*(1.0-left_leg_mask)

        return [Tensor(left_leg_img).reshape((1,*left_leg_img.shape))]

NODE_CLASS_MAPPINGS = {
    "OpenPoseWarp": OpenPoseWarp,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseWarp": "OpenPoseWarp",
}
