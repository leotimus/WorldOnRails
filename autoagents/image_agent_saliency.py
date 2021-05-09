class ImageAgentSaliency:
    def __init__(self, wide_rgb, cmd_value, steer_logits, throt_logits, brake_logits):
        self.wide_rgb = wide_rgb
        self.cmd_value = cmd_value
        self.steer_logits = steer_logits
        self.throt_logits = throt_logits
        self.brake_logits = brake_logits