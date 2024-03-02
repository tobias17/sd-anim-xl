from torch import Tensor
import numpy as np
import cv2
from typing import Tuple

hue_index = {
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
                "body_mask":      ("IMAGE",),
                "right_arm_mask": ("IMAGE",),
                "left_arm_mask":  ("IMAGE",),
                "right_leg_mask": ("IMAGE",),
                "left_leg_mask":  ("IMAGE",),
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
        return self.scale_point(min(where[1]), min(where[0])), self.scale_point(max(where[1]), max(where[0]))

    def open_pose_warp(self, stretch_image:Tensor, stretch_pose:Tensor, body_mask:Tensor, right_arm_mask:Tensor, left_arm_mask:Tensor, right_leg_mask:Tensor, left_leg_mask:Tensor, target_pose:Tensor):
        assert stretch_image.shape[0] == 1 and stretch_pose.shape[0] == 1, "cannot have a batch larger than 1 for stretch image and pose"
        stretch_image = stretch_image[0].numpy()
        stretch_pose  = stretch_pose[0].numpy()
        if stretch_pose.shape != stretch_image.shape:
            self.rescale = list(stretch_image.shape[i] / stretch_pose.shape[i] for i in range(2))
        else:
            self.rescale = (1.0,1.0)

        hsv_pose_img = cv2.cvtColor(stretch_pose, cv2.COLOR_BGR2HSV)

        p1, p2 = self.extract_position(hsv_pose_img, 'leg_right_upper')
        print(p1, p2)

        stretch_image = stretch_image
        cv2.circle(stretch_image, p1, 5, (0,0,255), 8)
        cv2.circle(stretch_image, p2, 5, (0,255,0), 8)

        return [Tensor(stretch_image).reshape((1,*stretch_image.shape))]

NODE_CLASS_MAPPINGS = {
    "OpenPoseWarp": OpenPoseWarp,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseWarp": "OpenPoseWarp",
}
