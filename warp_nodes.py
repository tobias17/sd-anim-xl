from torch import Tensor
import numpy as np
import cv2

class OpenPoseWarp:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "stretch_image":  ("IMAGE",),
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

    def open_pose_warp(self, stretch_image:Tensor, body_mask:Tensor, right_arm_mask:Tensor, left_arm_mask:Tensor, right_leg_mask:Tensor, left_leg_mask:Tensor, target_pose:Tensor):
        assert stretch_image.shape[0] == 1, "cannot have a batch larger than 1 right now"
        hsv_image = cv2.cvtColor(stretch_image[0].numpy(), cv2.COLOR_BGR2HSV)
        print(hsv_image)
        mask = cv2.inRange(hsv_image, np.array([215,0.9,0.5]), np.array([225,1.1,0.7]))

        return [Tensor(mask).reshape((1,*mask.shape,1)).expand((1,*mask.shape,3))]

NODE_CLASS_MAPPINGS = {
    "OpenPoseWarp": OpenPoseWarp,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseWarp": "OpenPoseWarp",
}
