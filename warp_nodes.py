from torch import Tensor
import numpy as np
import cv2
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

    def open_pose_warp(self, stretch_image:Tensor, stretch_pose:Tensor, body_mask:Tensor, right_arm_mask:Tensor, left_arm_mask:Tensor, right_leg_mask:Tensor, left_leg_mask:Tensor, target_pose:Tensor):
        assert stretch_image.shape[0] == 1 and stretch_pose.shape[0] == 1, "cannot have a batch larger than 1 for stretch image and pose"
        stretch_image = stretch_image[0].numpy()
        stretch_pose  = stretch_pose[0].numpy()
        if stretch_pose.shape != stretch_image.shape:
            self.rescale = list(stretch_image.shape[i] / stretch_pose.shape[i] for i in range(2))
        else:
            self.rescale = (1.0,1.0)
        
        print(right_arm_mask)
        print(type(right_arm_mask))
        print(right_arm_mask.shape)

        hsv_pose_img = cv2.cvtColor(stretch_pose, cv2.COLOR_BGR2HSV)
        p1, p2 = self.extract_position(hsv_pose_img, 'leg_right_upper')

        stretch_image = stretch_image
        cv2.circle(stretch_image, p1, 3, (0,0,255), 8)
        cv2.circle(stretch_image, p2, 3, (0,255,0), 8)

        return [Tensor(stretch_image).reshape((1,*stretch_image.shape))]

NODE_CLASS_MAPPINGS = {
    "OpenPoseWarp": OpenPoseWarp,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseWarp": "OpenPoseWarp",
}
