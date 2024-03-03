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

def overlay_images(bg, fg):
    abg = bg[:,:,3]/1.0
    afg = fg[:,:,3]/1.0
    for c in range(3):
        bg[:,:,c] = afg * fg[:,:,c]  +  abg * bg[:,:,c] * (1-afg)
    bg[:,:,3] = (1 - (1 - afg) * (1 - abg)) * 1.0
    return bg

def to_circle(angle, normalize_to=0) -> float:
    while angle >  math.pi + normalize_to:  angle -= 2*math.pi
    while angle < -math.pi + normalize_to:  angle += 2*math.pi
    return angle

class Vector2:
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y
    def point(self) -> Tuple[float,float]: return (self.x, self.y)
    def point_int(self) -> Tuple[int,int]: return (int(self.x), int(self.y))
    def __sub__(self, o) -> 'Vector2': return Vector2(self.x - o.x, self.y - o.y)
    def __add__(self, o) -> 'Vector2': return Vector2(self.x + o.x, self.y + o.y)
    def angle(self) -> float: return math.atan2(self.y, self.x)
    def length(self) -> float: return math.sqrt(self.x**2 + self.y**2)
    def __str__(self) -> str: return f"Point(x={self.x},y={self.y})"

    def __mul__(self, o) -> 'Vector2':
        if isinstance(o, type(self)): return Vector2(self.x * o.x, self.y * o.y)
        else:                         return Vector2(self.x * o,   self.y * o  )

    def normalize(self, length=1) -> 'Vector2':
        assert length != 0, "Can not normalize a vector to length 0!"
        assert self.x != 0 or self.y != 0, "Can not normalize a vector of length 0!"
        div = self.length() / float(length)
        return Vector2(self.x / div, self.y / div)

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
        move_vec = Vector2(math.cos(start_angle*math.pi/180), math.sin(start_angle*math.pi/180))
        for i in range(1, 50+1):
            point = (root + (move_vec * i)).point_int()
            color = list(ref_img[point[1], point[0]])
            cv2.ellipse(draw_img, root.point_int(), (i,i), 0, start_angle, end_angle, color, 2)

    def get_transform_points(self):
        points = []
        for is_left in [True, False]:
            for is_first in [True, False]:
                dir_vec = (self.p1 - self.p2 if is_first else self.p2 - self.p1).normalize(self.transform_offset)
                dir_vec = Vector2((1 if is_left else -1) * dir_vec.y, (-1 if is_left else 1) * dir_vec.x)
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
        self.vec = Vector2(math.cos(self.angle), math.sin(self.angle))
    
    def draw_mask_on(self, img, color):
        angle = self.angle
        mod_dist = Vector2(math.cos(angle - math.pi/2) * self.dist, math.sin(angle - math.pi/2) * self.dist)
        cv2.line(img, (self.comp1.p2 + mod_dist - self.vec*self.dist).point_int(), (self.comp1.p2 + mod_dist + self.vec*self.dist).point_int(), color, int(self.dist*2))

    def draw_debug_on(self, img, p1, length, color, size):
        cv2.line(img, (p1 + (self.vec * length)).point_int(), (p1 - (self.vec * length)).point_int(), color, size)
        cv2.circle(img, (p1 + (self.vec * length)).point_int(), 5, (0, 0, 1, 1), -1)

class OpenPoseWarp:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "stretch_image":  ("IMAGE",),
                "stretch_pose":   ("IMAGE",),
                "stretch_mask":   ("MASK",),
                "mask_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "body_mask":      ("MASK",),
                "right_arm_mask": ("MASK",),
                "left_arm_mask":  ("MASK",),
                "right_leg_mask": ("MASK",),
                "left_leg_mask":  ("MASK",),
                "target_pose":    ("IMAGE",),
                "debug":          ("BOOLEAN", {"default": False}),
            }
        }
    
    CATEGORY = "PoseWarp"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("target_pose",)
    FUNCTION = "open_pose_warp"

    def extract_points(self, hsv_pose_img:np.ndarray, body_parts:str) -> Tuple[Vector2,Vector2]:
        p1, p2 = Vector2(0,0), Vector2(0,0)
        for body_part in body_parts:
            hue = hue_index.get(body_part, None)
            assert hue is not None, f"failed to find body part key '{body_part}' in hue_index dict"
            mask = cv2.inRange(hsv_pose_img, np.array([hue-2,0.98,0.58]), np.array([hue+2,1.02,0.62]))
            where = np.where(mask > 200)

            x1_h, x2_h = min(where[1]), max(where[1])
            y1_h, y2_h = min(np.where(mask[:,x1_h:x1_h+2] > 200)[0]), max(np.where(mask[:,x2_h-1:x2_h+1] > 200)[0])

            y1_v, y2_v = min(where[0]), max(where[0])
            x1_v, x2_v = min(np.where(mask[y1_v:y1_v+2,:] > 200)[1]), max(np.where(mask[y2_v-1:y2_v+1,:] > 200)[1])

            if ((x2_h-x1_h)**2 + (y2_h-y1_h)**2) > ((x2_v-x1_v)**2 + (y2_v-y1_v)**2):
                p1 += Vector2(x1_h, y1_h)*self.rescale
                p2 += Vector2(x2_h, y2_h)*self.rescale
            else:
                p1 += Vector2(x1_v, y1_v)*self.rescale
                p2 += Vector2(x2_v, y2_v)*self.rescale
        return p1 * (1/len(body_parts)), p2 * (1/len(body_parts))

    def extract_comps_and_bound(self, hsv_pose_img:np.ndarray, keys1:str, keys2:str) -> Tuple[TransformComponent,TransformComponent,TransformBoundary]:
        p1, p2 = self.extract_points(hsv_pose_img, keys1)
        p3, p4 = self.extract_points(hsv_pose_img, keys2)
        dists = [
            ((p1-p3).length(), lambda: (p2,p1,p3,p4)),
            ((p1-p4).length(), lambda: (p2,p1,p4,p3)),
            ((p2-p3).length(), lambda: (p1,p2,p3,p4)),
            ((p2-p4).length(), lambda: (p1,p2,p4,p3)),
        ]
        dists = sorted(dists, key=lambda v: v[0])
        p1,p2,p3,p4 = dists[0][1]()
        p2 = p3 = (p2+p3)*0.5
        c1,c2 = TransformComponent(p1,p2), TransformComponent(p3,p4)
        return c1, c2, TransformBoundary(c1,c2)

    def clean_mask(self, mask:Tensor) -> np.ndarray:
        assert mask.shape[0], f"masks must be of batch size 1, found {mask.shape[0]}"
        mask_np = mask[0].numpy()
        return mask_np.reshape((*mask_np.shape,1)).copy()

    def open_pose_warp(self, stretch_image:Tensor, stretch_pose:Tensor, stretch_mask:Tensor, mask_threshold:float, body_mask:Tensor, right_arm_mask:Tensor, left_arm_mask:Tensor, right_leg_mask:Tensor, left_leg_mask:Tensor, target_pose:Tensor, debug:bool):
        assert stretch_image.shape[0] == 1 and stretch_pose.shape[0] == 1, "cannot have a batch larger than 1 for stretch image and pose"
        stretch_image_np: np.ndarray = stretch_image[0].numpy()
        stretch_pose_np:  np.ndarray = stretch_pose[0].numpy()
        target_pose_np:   np.ndarray = target_pose[0].numpy()
        if stretch_pose_np.shape != stretch_image_np.shape:
            r = list(stretch_image_np.shape[i] / stretch_pose_np.shape[i] for i in range(2))
        else:
            r = (1.0,1.0)
        self.rescale = Vector2(r[1], r[0])

        images = []
        full_img = np.ones((*stretch_image_np.shape[:-1],4)) * (0.5,0.5,0.5,1.0)

        stretch_mask_np = self.clean_mask(stretch_mask)
        stretch_mask_np[stretch_mask_np < mask_threshold] = 0.0
        stretch_mask_np[stretch_mask_np > mask_threshold] = 1.0
        if debug: images.append(stretch_mask_np.copy())

        body_mask_np      = self.clean_mask(body_mask)      * stretch_mask_np
        right_arm_mask_np = self.clean_mask(right_arm_mask) * stretch_mask_np
        left_arm_mask_np  = self.clean_mask(left_arm_mask)  * stretch_mask_np
        right_leg_mask_np = self.clean_mask(right_leg_mask) * stretch_mask_np
        left_leg_mask_np  = self.clean_mask(left_leg_mask)  * stretch_mask_np

        loop = [
            (left_arm_mask_np,  ('arm_left_upper',),  ('arm_left_lower',),  0.7),
            (left_leg_mask_np,  ('leg_left_upper',),  ('leg_left_lower',),  0.7),
            (body_mask_np,      ('head',),   ('torso_left','torso_right',), 1.0),
            (right_leg_mask_np, ('leg_right_upper',), ('leg_right_lower',), 1.0),
            (right_arm_mask_np, ('arm_right_upper',), ('arm_right_lower',), 1.0),
        ]
        for mask_np, keys1, keys2, scale in loop:
            input_img = np.ones((*stretch_image_np.shape[:-1],4))
            input_img[:,:,:3] = stretch_image_np
            input_img[:,:,3:] = mask_np

            hsv_source_pose = cv2.cvtColor(stretch_pose_np, cv2.COLOR_BGR2HSV)
            hsv_target_pose = cv2.cvtColor(target_pose_np, cv2.COLOR_BGR2HSV)
            s_comp1, s_comp2, s_bound = self.extract_comps_and_bound(hsv_source_pose, keys1, keys2)
            e_comp1, e_comp2, e_bound = self.extract_comps_and_bound(hsv_target_pose, keys1, keys2)

            mask = np.zeros(input_img.shape[:-1])
            s_bound.draw_mask_on(mask, (1.0,))

            upper_img = input_img.copy()
            upper_img[mask > 0.5] = (0,0,0,0)
            lower_img = input_img.copy()
            lower_img[mask < 0.5] = (0,0,0,0)
            s_comp2.draw_boundary_on(lower_img, input_img, e_bound.diff*180/math.pi*2.5)

            size = (input_img.shape[1], input_img.shape[0],)
            upper_img = s_comp1.transform_image(upper_img, e_comp1, size)
            lower_img = s_comp2.transform_image(lower_img, e_comp2, size)

            comb_img = lower_img.copy()
            overlay_images(comb_img, upper_img)
            comb_img *= (scale,scale,scale,1.0)
            overlay_images(full_img, comb_img)

            if debug:
                background_img = np.ones((*input_img.shape[:-1],3)) * 0.2
                for loop_img in [input_img, upper_img, lower_img, comb_img]:
                    loop_img = loop_img.copy()
                    db_img = loop_img[:,:,:3]*loop_img[:,:,3:] + background_img*(1.0-loop_img[:,:,3:])
                    s_comp1.draw_debug_on(db_img, (0,0,1,1))
                    s_comp2.draw_debug_on(db_img, (.3,.3,1,1))
                    s_bound.draw_debug_on(db_img, s_comp1.p2, 20, (1,0,0,1), 4)
                    e_comp1.draw_debug_on(db_img, (0,1,0,1))
                    e_comp2.draw_debug_on(db_img, (.3,1,.3,1))
                    e_bound.draw_debug_on(db_img, e_comp1.p2, 20, (1,0,0,1), 4)
                    images.append(db_img[:,:,:3])

        images = [full_img[:,:,:3]] + images
        output = np.ones((len(images),*images[0].shape))
        for i in range(len(images)):
            output[i] *= images[i]
        return [Tensor(output)]

NODE_CLASS_MAPPINGS = {
    "OpenPoseWarp": OpenPoseWarp,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseWarp": "OpenPoseWarp",
}
