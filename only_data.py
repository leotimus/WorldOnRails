import torch
from torch import nn
from common.resnet import resnet34
from common.normalize import Normalize
from rails.models.main_model import action_logits


def action_prob(steer_logit, throt_logit, brake_logit):
    steer_logit = steer_logit.repeat(self.num_throts)
    throt_logit = throt_logit.repeat_interleave(self.num_steers)

    action_logit = torch.cat([steer_logit, throt_logit, brake_logit[None]])

    return torch.softmax(action_logit, dim=0)


def main():
    torch.device('cuda')
    wide_rgb = torch.load('expirements/wire_rgb_1620392696025.pt').cuda()
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device='cuda')
    backbone_wide = resnet34(pretrained=True).cuda()
    wide_embed = backbone_wide(normalize(wide_rgb / 255.))
    embed = wide_embed.mean(dim=[2, 3])
    act_head = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, 256),
        nn.ReLU(True),
        nn.Linear(256, 312),
    ).cuda()
    num_cmds = 6
    num_speeds = 4
    num_steers = 9
    num_throts = 3
    cmd = 3
    act_output = act_head(embed).view(-1, num_cmds, num_speeds, num_steers + num_throts + 1)
    act_output = action_logits(act_output, num_steers, num_throts)
    # Action logits
    steer_logits = act_output[0, cmd, :, :9]
    throt_logits = act_output[0, cmd, :, 9:12]
    brake_logits = act_output[0, cmd, :, -1]
    print(f'steer_logits = {steer_logits}, throt_logits = {throt_logits}, brake_logits = {brake_logits}')

    # action_proba = action_prob(steer_logit, throt_logit, brake_logit)
    # brake_prob = float(action_prob[-1])
    #
    # steer = float(self.steers @ torch.softmax(steer_logit, dim=0))
    # throt = float(self.throts @ torch.softmax(throt_logit, dim=0))
    #
    # steer, throt, brake = post_process(steer, throt, brake_prob, spd, cmd_value)

main()