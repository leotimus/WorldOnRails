import os, math
from functools import partial
import pytz
from datetime import datetime
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
import random
import cv2, time
import traceback
from multiprocessing.pool import ThreadPool
from itertools import repeat

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from agents.navigation.local_planner import RoadOption
from autoagents.image_agent_saliency import ImageAgentSaliency
from utils import visualize_obs

from rails.models import EgoModel, CameraModel
from autoagents.waypointer import Waypointer

from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize
import skimage
import skimage.io
from skimage.util import img_as_float, img_as_ubyte
import matplotlib as mpl ; mpl.use("Agg")
import matplotlib.animation as manimation
import matplotlib.pyplot as plt

def get_entry_point():
    return 'ImageAgent'


def create_and_save_saliency(image_agent, video_saliency, video_original, info):
    saliency = image_agent.score_frame(info, density=10, radius=10)
    cv2.imwrite(f'experiments/saliency_{get_time_mils()}.png', saliency)
    orig_img = info.wide_rgb.copy()
    saliency_img = image_agent.apply_saliency(saliency, info.wide_rgb, channel=0)
    video_saliency.write(saliency_img)
    video_original.write(orig_img)

def create_and_save_saliency_ffmpeg(image_agent, info):
    saliency = image_agent.score_frame(info, density=10, radius=10)
    saliency_img_throttle = image_agent.apply_saliency(saliency[0], info.wide_rgb, channel=0)
    saliency_img_brake = image_agent.apply_saliency(saliency[1], info.wide_rgb, channel=1)
    saliency_img_steer = image_agent.apply_saliency(saliency[2], info.wide_rgb, channel=2)
    return saliency_img_throttle, saliency_img_brake, saliency_img_steer


def get_time_mils():
    return int(round(time.time() * 1000))


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
        mask = self.get_mask(center, size, radius)
        occlude = lambda I, mask: I * (1 - mask) + gaussian_filter(I, sigma=3) * mask
        im = occlude(wide_rgb.squeeze(), mask).reshape(240, 480)#(3, 480, 240) --> ValueError: operands could not be broadcast together with shapes (240,480,3) (480,240)
        return im

    def get_mask(self, center, size, radius):
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        mask = gaussian_filter(mask, sigma=radius)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
        return mask / mask.max()

    def score_frame(self, saliencyInfo, density=10, radius=5):
        # r: radius of blur
        # d: density of scores (if d==1, then get a score for every pixel...
        #    if d==2 then every other, which is 25% of total pixels for a 2D image)
        wide_rgb = img_as_float(skimage.color.rgb2gray(saliencyInfo.wide_rgb))#Converting to gray picture to be normalize and 2d arrau
        steer_Logits = saliencyInfo.steer_logits
        throt_Logits = saliencyInfo.throt_logits
        brake_Logits = saliencyInfo.brake_logits
        cmd_value = saliencyInfo.cmd_value

        scores_throttle = np.zeros((int(240 / density) + 1, int(480 / density) + 1))  # saliency scores S(t,i,j)
        scores_steer = np.zeros((int(240 / density) + 1, int(480 / density) + 1))  # saliency scores S(t,i,j)
        scores_brake = np.zeros((int(240 / density) + 1, int(480 / density) + 1))  # saliency scores S(t,i,j)
        for i in range(0, 240, density):
            for j in range(0, 480, density):
                masking_wide_rgp = self.create_masking(wide_rgb, center=[i, j], size=[240,  480], radius=radius)

                masking_wide_rgp = skimage.color.gray2rgb(masking_wide_rgp)
                masking_wide_rgp = img_as_ubyte(masking_wide_rgp)

                _masking_wide_rgp = masking_wide_rgp[self.wide_crop_top:, :, :3]
                _masking_wide_rgp = _masking_wide_rgp[..., ::-1].copy()
                _masking_wide_rgp = torch.tensor(_masking_wide_rgp[None]).float().permute(0, 3, 1, 2).to(self.device)
                steer_logits, throt_logits, brake_logits = self.image_model.policy(_masking_wide_rgp, None, cmd_value)

                x = int(i / density)
                y = int(j / density)
                current_score_steer = (steer_Logits - steer_logits).pow(2).sum().mul_(.5)
                current_score_throttle = (throt_Logits - throt_logits).pow(2).sum().mul_(.5)
                current_score_brake = (brake_Logits - brake_logits).pow(2).sum().mul_(.5)
                scores_throttle[x, y] = current_score_throttle
                scores_steer[x, y] = current_score_steer
                scores_brake[x, y] = current_score_brake



        pmax_throttle = scores_throttle.max()
        scores_throttle = imresize(scores_throttle, size=[240, 480], interp='lanczos').astype(np.float32)
        res_throttle = pmax_throttle * scores_throttle / scores_throttle.max()

        pmax_brake = scores_brake.max()
        scores_brake = imresize(scores_brake, size=[240, 480], interp='lanczos').astype(np.float32)
        res_brake = pmax_brake * scores_brake / scores_brake.max()

        pmax_steer = scores_steer .max()
        scores_steer = imresize(scores_steer , size=[240, 480], interp='lanczos').astype(np.float32)
        res_steer = pmax_steer  * scores_steer  / scores_steer.max()

        # res = res/100.

        log = get_time_mils()
        tz = pytz.timezone('Europe/Berlin')
        time_stamp = str(datetime.now(tz))
        score_img_name_throttle = f'experiments/scores_throttle_{log}_{time_stamp}_lanczos.png'
        score_img_name_brake = f'experiments/scores_brake_{log}_{time_stamp}_lanczos.png'
        score_img_name_steer = f'experiments/scores_steer_{log}_{time_stamp}_lanczos.png'
        #res_img_name_throttle = f'experiments/res_{log}_{time_stamp}_lanczos.png'
        cv2.imwrite(score_img_name_throttle, scores_throttle)
        cv2.imwrite(score_img_name_brake, scores_brake)
        cv2.imwrite(score_img_name_steer, scores_steer)
        #cv2.imwrite(res_img_name_throttle, res_throttle)
        score_image_throttle = cv2.imread(score_img_name_throttle)
        score_image_brake = cv2.imread(score_img_name_brake)
        score_image_steer = cv2.imread(score_img_name_steer)
        scores_denoised_throttle = cv2.fastNlMeansDenoising(score_image_throttle, None, 10, 7, 21)
        scores_denoised_brake = cv2.fastNlMeansDenoising(score_image_brake, None, 10, 7, 21)
        scores_denoised_steer = cv2.fastNlMeansDenoising(score_image_steer, None, 10, 7, 21)
        movsd_throttle = np.argmax(np.bincount(scores_denoised_throttle.flat))
        movsd_brake = np.argmax(np.bincount(scores_denoised_brake.flat))
        movsd_steer = np.argmax(np.bincount(scores_denoised_steer.flat))
        erased_gray_score_throttle = np.where(scores_denoised_throttle <= movsd_throttle + 55, 0, scores_denoised_throttle)
        erased_gray_score_throttle = skimage.color.rgb2gray(erased_gray_score_throttle)
        new_res_throttle = pmax_throttle * erased_gray_score_throttle / erased_gray_score_throttle.max()

        erased_gray_score_brake = np.where(scores_denoised_brake <= movsd_brake + 55, 0,
                                              scores_denoised_brake)
        erased_gray_score_brake = skimage.color.rgb2gray(erased_gray_score_brake)
        new_res_brake = pmax_brake* erased_gray_score_brake / erased_gray_score_brake.max()

        erased_gray_score_steer = np.where(scores_denoised_steer <= movsd_steer + 55, 0,
                                              scores_denoised_steer)
        erased_gray_score_steer = skimage.color.rgb2gray(erased_gray_score_steer)
        new_res_steer = pmax_steer * erased_gray_score_steer / erased_gray_score_steer.max()
        #cv2.imwrite(f'experiments/scores_denoised_{log}_{time_stamp}_lanczos.png', scores_denoised_throttle)
        #cv2.imwrite(f'experiments/scores_denoised_outgrayed_larger_scale_{log}_{time_stamp}_lanczos.png', erased_gray_score_throttle)
        #cv2.imwrite(f'experiments/res_denoised_outgrayed_larger_scale{log}_{time_stamp}_lanczos.png', new_res_throttle)
        return [new_res_throttle, new_res_brake, new_res_steer]

    def apply_saliency(self, saliency, frame, fudge_factor=400, channel=0, sigma=0):
        # sometimes saliency maps are a bit clearer if you blur them
        # slightly...sigma adjusts the radius of that blur
        pmax = saliency.max()
        S = imresize(saliency, size=[240, 480], interp='lanczos').astype(np.float32)#Double it like in origian;
        S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
        S -= S.min()
        S = fudge_factor * pmax * S / S.max()
        I = frame.astype('uint16')
        I[:, :,channel] += S.astype('uint16')
        I = I.clip(1, 255).astype('uint8')
        return I

    def saveSaliencyVideoFFMpeg(self, Ls):
        tz = pytz.timezone('Europe/Berlin')
        time_stamp = str(datetime.now(tz))
        start = time.time()
        movie_title_saliency = "original_saliency_compare_video_{}_{}.mp4".format(int(round(time.time() * 1000)), time_stamp) #f'experiments/original_throttle_{int(round(time.time() * 1000))}_video_{time_stamp}.avi'
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=movie_title_saliency, artist='greydanus', comment='atari-saliency-video')
        writer = FFMpegWriter(fps=8, metadata=metadata)
        prog = '';
        #Setup for Original grame and underneath Saliency Frame of one feature
        """
        f, ax = plt.subplots(2, figsize=[6, 6 * 1.3], dpi=75)
        with writer.saving(f, "experiments/" + movie_title_saliency, 75):
            for s in Ls:
                saliency_frame = create_and_save_saliency_ffmpeg(self, s)
                ax[0].imshow(cv2.cvtColor(s.wide_rgb, cv2.COLOR_BGR2RGB))
                ax[0].set_title('Original Frame')
                ax[1].imshow(cv2.cvtColor(saliency_frame, cv2.COLOR_BGR2RGB))
                ax[1].set_title('Saliency Frame')
                writer.grab_frame()
                tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                print('\ttime: {}'.format(tstr), end='\r')
        print('\nfinished.')
        """
        f, ax= plt.subplots(2,2)
        f.tight_layout()
        with writer.saving(f, "experiments/" + movie_title_saliency, 75):
            for s in Ls:
                s_throttle, s_brake, s_steer = create_and_save_saliency_ffmpeg(self, s)
                ax[0,0].imshow(cv2.cvtColor(s.wide_rgb, cv2.COLOR_BGR2RGB))
                ax[0,0].set_title('Original Frame')
                ax[0,0].set_aspect('equal')
                ax[0,1].imshow(cv2.cvtColor(s_throttle, cv2.COLOR_BGR2RGB))
                ax[0,1].set_title('Saliency Frame Throttle')
                ax[0,1].set_aspect('equal')
                ax[1,0].imshow(cv2.cvtColor(s_brake, cv2.COLOR_BGR2RGB))
                ax[1,0].set_title('Saliency Frame Brake')
                ax[1,0].set_aspect('equal')
                ax[1,1].imshow(cv2.cvtColor(s_steer, cv2.COLOR_BGR2RGB))
                ax[1,1].set_title('Saliency Frame Steer')
                ax[1,1].set_aspect('equal')
                writer.grab_frame()
                tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                print('\ttime: {}'.format(tstr), end='\r')
        print('\nfinished.')

    def saveSaliencyVideo(self, Ls):
        try:
            print('log videos....')
            start = time.time()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            tz = pytz.timezone('Europe/Berlin')
            time_stamp = str(datetime.now(tz))
            video_saliency = cv2.VideoWriter(
                f'experiments/saliency_throttle_{get_time_mils()}_video_{time_stamp}.avi', fourcc, 1, (480, 240))
            video_original = cv2.VideoWriter(
                f'experiments/original_throttle_{get_time_mils()}_video_{time_stamp}.avi', fourcc, 2, (480, 240))
            # torch.save(self.Ls, f'expirements/flush_{int(round(time.time() * 1000))}.data')

            pool = ThreadPool(processes=2)
            pool.starmap(create_and_save_saliency, zip(repeat(self), repeat(video_saliency), repeat(video_original),  Ls))
            pool.close()
            pool.terminate()
            pool.join()
            """
            for s in Ls:
                create_and_save_saliency(self, video, s)
            """
            tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
            print('\ttime: {}'.format(tstr), end='\r')
            cv2.destroyAllWindows()
            video_saliency.release()
            video_original.release()
            Ls.clear()
        except:
           traceback.print_exc()

    def save_input_sensor_video(self):
        print("Save Input")
        torch.save(self.Ls, f'experiments/new_flush_{int(round(time.time() * 1000))}.data')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f'experiments/video_stuff_{get_time_mils()}.avi', fourcc, 1, (480, 240))
        for frame in self.inputCamera:
            video.write(frame)
        print("release")
        cv2.destroyAllWindows()
        video.release()
        self.inputCamera.clear()

    def flush_data(self):
        #self.save_input_sensor_video()
        self.saveSaliencyVideo(self.Ls)

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })

        self.vizs.clear()

    def analzye_data(self, Ls):
        highest_brake = {"mean": 0, "index":0}
        lowest_brake = {"mean": 1000, "index":0}
        for index, s in enumerate(Ls):
            if torch.mean(s.brake_logits) > highest_brake["mean"]:
                highest_brake["mean"] = torch.mean(s.brake_logits)
                highest_brake["index"] = index
            if torch.mean(s.brake_logits) < lowest_brake["mean"]:
                lowest_brake["mean"] = torch.mean(s.brake_logits)
                lowest_brake["index"] = index
        print(highest_brake )
        print(lowest_brake)

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
        _wide_rgb = _wide_rgb[..., ::-1].copy()
        _wide_rgb = torch.tensor(_wide_rgb[None]).float().permute(0, 3, 1, 2).to(self.device)
        _narr_rgb = narr_rgb[:-self.narr_crop_bottom,:,:3]
        _narr_rgb = _narr_rgb[...,::-1].copy()
        _narr_rgb = torch.tensor(_narr_rgb[None]).float().permute(0, 3, 1, 2).to(self.device)

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
