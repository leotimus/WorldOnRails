import os
import math
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
import random
import cv2, time

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from agents.navigation.local_planner import RoadOption
from autoagents.image_agent_saliency import ImageAgentSaliency
from utils import visualize_obs

from rails.models import EgoModel, CameraModel
from autoagents.waypointer import Waypointer

from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize


def get_entry_point():
    return 'ImageAgent'

class ImageAgent(AutonomousAgent):
    
    """
    Trained image agent
    """
    
    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.track = Track.SENSORS
        self.num_frames = 0

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        self.device = torch.device('cuda')

        self.image_model = CameraModel(config).to(self.device)
        self.image_model.load_state_dict(torch.load(self.main_model_dir))
        self.image_model.eval()

        self.vizs = []

        self.waypointer = None

        if self.log_wandb:
            wandb.init(project='carla_evaluate')
            
        self.steers = torch.tensor(np.linspace(-self.max_steers,self.max_steers,self.num_steers)).float().to(self.device)
        self.throts = torch.tensor(np.linspace(0,self.max_throts,self.num_throts)).float().to(self.device)

        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0

        self.inputCamera = []
        self.Ls = []

    def destroy(self):
        if len(self.vizs) == 0:
            return

        self.flush_data()
        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0
        self.lane_changed = None
        
        del self.waypointer
        del self.image_model

    def create_masking(self, wide_rgb, center, size, radius):
        # prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80)
        mask = self.get_mask(center, size, radius)
        occlude = lambda I, mask: I * (1 - mask) + gaussian_filter(I, sigma=3) * mask
        im = occlude(wide_rgb.squeeze(), mask).reshape(3, 480, 240)
        return im

    def get_mask(self, center, size, radius):
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        mask = gaussian_filter(mask, sigma=radius)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
        return mask / mask.max()

    def score_frame(self, saliencyInfo, density=5, radius=5):
        # r: radius of blur
        # d: density of scores (if d==1, then get a score for every pixel...
        #    if d==2 then every other, which is 25% of total pixels for a 2D image)

        wide_rgb = saliencyInfo.wide_rgb
        steer_Logits = saliencyInfo.steer_logits
        throt_Logits = saliencyInfo.throt_logits
        brake_Logits = saliencyInfo.brake_logits
        cmd_value = saliencyInfo.cmd_value

        scores = np.zeros((int(480 / density) + 1, int(240 / density) + 1))  # saliency scores S(t,i,j)
        for i in range(0, 480, density):
            for j in range(0, 240, density):
                masking_wide_rgp = self.create_masking(wide_rgb, center=[i, j], size=[480, 240], radius=radius)
                steer_logits, throt_logits, brake_logits = self.image_model.policy(masking_wide_rgp, None, cmd_value)
                scores[int(i / wide_rgb), int(j / wide_rgb)] = (brake_Logits - brake_logits).pow(2).sum().mul_(.5).data[0]
        pmax = scores.max()
        scores = imresize(scores, size=[480, 240], interp='bilinear').astype(np.float32)
        return pmax * scores / scores.max()

    def apply_saliency(self, saliency, frame, fudge_factor=400, channel=2, sigma=0):
        # sometimes saliency maps are a bit clearer if you blur them
        # slightly...sigma adjusts the radius of that blur
        pmax = saliency.max()
        S = imresize(saliency, size=[480, 240], interp='bilinear').astype(np.float32)
        S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
        S -= S.min()
        S = fudge_factor * pmax * S / S.max()
        I = frame.astype('uint16')
        I[35:195, :, channel] += S.astype('uint16')
        I = I.clip(1, 255).astype('uint8')
        return I

    def saveSaliencyVideo(self, Ls):
        try:
            print('log videos....')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(f'expirements/saliency_{int(round(time.time() * 1000))}.avi', fourcc, 1, (480, 240))
            # torch.save(self.Ls, f'expirements/flush_{int(round(time.time() * 1000))}.data')

            for saliencyInfo in Ls:
                saliency = self.score_frame(saliencyInfo)
                img = self.apply_saliency(saliency, saliencyInfo.wide_rgb)
                video.write(img)

            cv2.destroyAllWindows()
            video.release()
            Ls.clear()
        except:
            print('false to save saliency video')

    def save_input_sensor_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4')
        video = cv2.VideoWriter(f'expirements/{int(round(time.time() * 1000))}.avi', fourcc, 1, (480, 240))
        for frame in self.inputCamera:
            video.write(frame)
        cv2.destroyAllWindows()
        video.release()
        self.inputCamera.clear()

    def flush_data(self):
        # self.save_input_sensor_video()
        self.saveSaliencyVideo(self.Ls)

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })
            
        self.vizs.clear()

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.stitch_camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_RGB'},
        ]
        
        return sensors

    def run_step(self, input_data, timestamp):
        
        _, wide_rgb = input_data.get(f'Wide_RGB')
        _, narr_rgb = input_data.get(f'Narrow_RGB')

        # cv2.imwrite(f'expirements/rgb/{int(round(time.time() * 1000))}.png', wide_rgb)
        # self.inputCamera.append(wide_rgb)

        # Crop images
        _wide_rgb = wide_rgb[self.wide_crop_top:,:,:3]
        _narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,:3]

        _wide_rgb = _wide_rgb[...,::-1].copy()
        _narr_rgb = _narr_rgb[...,::-1].copy()

        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')


        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)

        _, _, cmd = self.waypointer.tick(gps)

        spd = ego.get('spd')
        
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        if cmd_value in [4,5]:
            if self.lane_changed is not None and cmd_value != self.lane_changed:
                self.lane_change_counter = 0

            self.lane_change_counter += 1
            self.lane_changed = cmd_value if self.lane_change_counter > {4:200,5:200}.get(cmd_value) else None
        else:
            self.lane_change_counter = 0
            self.lane_changed = None

        if cmd_value == self.lane_changed:
            cmd_value = 3

        _wide_rgb = torch.tensor(_wide_rgb[None]).float().permute(0,3,1,2).to(self.device)
        _narr_rgb = torch.tensor(_narr_rgb[None]).float().permute(0,3,1,2).to(self.device)
        
        if self.all_speeds:
            steer_logits, throt_logits, brake_logits = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value)
            ias = ImageAgentSaliency(wide_rgb, cmd_value, steer_logits, throt_logits, brake_logits)
            self.Ls.append(ias)
            # Interpolate logits
            steer_logit = self._lerp(steer_logits, spd)
            throt_logit = self._lerp(throt_logits, spd)
            brake_logit = self._lerp(brake_logits, spd)
        else:
            steer_logit, throt_logit, brake_logit = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value, spd=torch.tensor([spd]).float().to(self.device))

        
        action_prob = self.action_prob(steer_logit, throt_logit, brake_logit)

        brake_prob = float(action_prob[-1])

        steer = float(self.steers @ torch.softmax(steer_logit, dim=0))
        throt = float(self.throts @ torch.softmax(throt_logit, dim=0))

        steer, throt, brake = self.post_process(steer, throt, brake_prob, spd, cmd_value)
        # print(f'Command = {RoadOption(cmd_value).name}, steer = {steer}, throt = {throt}, brake = {brake}')

        rgb = np.concatenate([wide_rgb, narr_rgb[...,:3]], axis=1)
        
        self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), spd, cmd=cmd_value+1))

        if len(self.vizs) > 1000:
            self.flush_data()

        self.num_frames += 1

        return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)
    
    def _lerp(self, v, x):
        D = v.shape[0]

        min_val = self.min_speeds
        max_val = self.max_speeds

        x = (x - min_val)/(max_val - min_val)*(D-1)

        x0, x1 = max(min(math.floor(x), D-1),0), max(min(math.ceil(x), D-1),0)
        w = x - x0

        return (1-w) * v[x0] + w * v[x1]

    def action_prob(self, steer_logit, throt_logit, brake_logit):

        steer_logit = steer_logit.repeat(self.num_throts)
        throt_logit = throt_logit.repeat_interleave(self.num_steers)

        action_logit = torch.cat([steer_logit, throt_logit, brake_logit[None]])

        return torch.softmax(action_logit, dim=0)

    def post_process(self, steer, throt, brake_prob, spd, cmd):
        
        if brake_prob > 0.5:
            steer, throt, brake = 0, 0, 1
        else:
            brake = 0
            throt = max(0.4, throt)

        # # To compensate for non-linearity of throttle<->acceleration
        # if throt > 0.1 and throt < 0.4:
        #     throt = 0.4
        # elif throt < 0.1 and brake_prob > 0.3:
        #     brake = 1

        if spd > {0:10,1:10}.get(cmd, 15)/3.6: # 10 km/h for turning, 15km/h elsewhere
            throt = 0

        # if cmd == 2:
        #     steer = min(max(steer, -0.2), 0.2)

        # if cmd in [4,5]:
        #     steer = min(max(steer, -0.4), 0.4) # no crazy steerings when lane changing

        return steer, throt, brake


def load_state_dict(model, path):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(path)
    
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
