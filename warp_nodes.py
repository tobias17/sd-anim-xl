from torch import Tensor

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
        

        return [stretch_image]

NODE_CLASS_MAPPINGS = {
    "OpenPoseWarp": OpenPoseWarp,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseWarp": "OpenPoseWarp",
}
