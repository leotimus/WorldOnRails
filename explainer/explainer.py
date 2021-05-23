import os
import time
from datetime import datetime
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool

import cv2
import numpy
import numpy as np
import pytz
import skimage
import skimage.color
import skimage.io
import torch
from matplotlib import animation, pyplot as plt
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter
from skimage import img_as_float, img_as_ubyte

from autoagents.image_agent import ImageAgent
from autoagents.image_agent_saliency import ImageAgentSaliency
from explainer.remote_explainer import RemoteExplainer
from explainer.utils import get_time_mils, logger, timestamp


class Explainer:
    def __init__(self, agent: ImageAgent, input_data: list, hosts: list):
        self.agent = agent
        self.input_data = input_data
        self.hosts = hosts
        self.explainers = Queue()
        for host in self.hosts:
            self.explainers.put(host)
        cpu_count = os.cpu_count()
        if cpu_count >= 8:
            num_threads = (cpu_count - 2) / 2
            logger.info(f'Setting pytorch to use {num_threads} cpus per process.')
            torch.set_num_threads(int(num_threads))

    def explain(self):
        # f'experiments/original_throttle_{int(round(time.time() * 1000))}_video_{time_stamp}.avi'
        logger.info(f'Explainer starts. Number of frame {len(self.input_data)}')
        movie_title_saliency = f'original_saliency_compare_video_{int(get_time_mils())}_{timestamp()}.mp4'
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title=movie_title_saliency, artist='Hephaestus', comment='carla saliency video.')
        writer = FFMpegWriter(fps=8, metadata=metadata)
        # Setup for Original grame and underneath Saliency Frame of one feature
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
        f, ax = plt.subplots(2, 2)
        f.tight_layout()

        pool_size = len(self.hosts) + 1
        logger.info(f'Create thread pool size {pool_size}.')
        pool = ThreadPool(processes=pool_size)
        results = []

        for i, s in enumerate(self.input_data):
            r = pool.apply_async(self.process_saliency_ffmpeg, args=(s, i,))
            results.append(r)
        logger.info(f'All jobs submitted, overlaying saliencies.')

        with writer.saving(f, "experiments/" + movie_title_saliency, 400):
            for i, r in enumerate(results):
                logger.info(f'process frame {i}...')
                s = self.input_data[i]
                s_throttle, s_brake, s_steer = r.get()
                ax[0, 0].imshow(cv2.cvtColor(s.wide_rgb, cv2.COLOR_BGR2RGB))
                ax[0, 0].set_title('Original Frame')
                ax[0, 0].set_aspect('equal')

                ax[0, 1].imshow(cv2.cvtColor(s_throttle, cv2.COLOR_BGR2RGB))
                ax[0, 1].set_title('Saliency Frame Throttle')
                ax[0, 1].set_aspect('equal')

                ax[1, 0].imshow(cv2.cvtColor(s_brake, cv2.COLOR_BGR2RGB))
                ax[1, 0].set_title('Saliency Frame Brake')
                ax[1, 0].set_aspect('equal')

                ax[1, 1].imshow(cv2.cvtColor(s_steer, cv2.COLOR_BGR2RGB))
                ax[1, 1].set_title('Saliency Frame Steer')
                ax[1, 1].set_aspect('equal')
                writer.grab_frame()

        logger.info('Explainer finished.')
        pool.close()
        pool.terminate()
        pool.join()

    def process_saliency_ffmpeg(self, s: ImageAgentSaliency, idx: int):
        explainer = self.explainers.get(block=True)
        try:
            if explainer == 'memory':
                logger.info(f'Submit frame {idx + 1} to local memory explainer.')
                throttle, brake, steer = self.create_and_save_saliency_ffmpeg(s, idx)
            else:
                logger.info(f'Submit frame {idx + 1} to remote explainer at {explainer}.')
                throttle, brake, steer = RemoteExplainer(explainer).create_and_save_saliency_ffmpeg(s, idx)
        except Exception as error:
            logger.error(f'Frame {idx} was broken due to error.', error)
            throttle = numpy.empty(shape=[4, 3])
            brake = numpy.empty(shape=[4])
            steer = numpy.empty(shape=[4, 9])
        self.explainers.put(explainer)
        return throttle, brake, steer

    def analzye_data(self, data):
        highest_brake = {"mean": 0, "index": 0}
        lowest_brake = {"mean": 1000, "index": 0}
        for index, s in enumerate(data):
            if torch.mean(s.brake_logits) > highest_brake["mean"]:
                highest_brake["mean"] = torch.mean(s.brake_logits)
                highest_brake["index"] = index
            if torch.mean(s.brake_logits) < lowest_brake["mean"]:
                lowest_brake["mean"] = torch.mean(s.brake_logits)
                lowest_brake["index"] = index
        logger.info(f'highest brake: {highest_brake}')
        logger.info(f'lowest_brake: {lowest_brake}')

    def create_and_save_saliency_ffmpeg(self, info: ImageAgentSaliency, i: int):
        logger.info(f'scoring frame {i + 1}')
        saliency = self.score_frame(info, density=10, radius=10)
        saliency_img_throttle = self.apply_saliency(saliency[0], info.wide_rgb, channel=0)
        saliency_img_brake = self.apply_saliency(saliency[1], info.wide_rgb, channel=1)
        saliency_img_steer = self.apply_saliency(saliency[2], info.wide_rgb, channel=2)
        logger.info(f'scored frame {i + 1}')
        return saliency_img_throttle, saliency_img_brake, saliency_img_steer

    def score_frame(self, saliency_info, density=10, radius=5):
        # r: radius of blur
        # d: density of scores (if d==1, then get a score for every pixel...
        #    if d==2 then every other, which is 25% of total pixels for a 2D image)
        # Converting to gray picture to be normalize and 2d array
        wide_rgb = saliency_info.wide_rgb.copy()
        # wide_rgb = imresize(wide_rgb, size=[120, 240], interp='lanczos')
        wide_rgb = img_as_float(skimage.color.rgb2gray(wide_rgb))
        steer_Logits = saliency_info.steer_logits
        throt_Logits = saliency_info.throt_logits
        brake_Logits = saliency_info.brake_logits
        cmd_value = saliency_info.cmd_value

        # down_scaled = imresize(wide_rgb, size=[120, 240], interp='lanczos')
        # cv2.imwrite(f'experiments/downscaled_{timestamp()}.png', down_scaled)

        target_height = 240
        target_width = 480
        scores_throttle = np.zeros(
            (int(target_height / density) + 1, int(target_width / density) + 1))  # saliency scores S(t,i,j)
        scores_steer = np.zeros(
            (int(target_height / density) + 1, int(target_width / density) + 1))  # saliency scores S(t,i,j)
        scores_brake = np.zeros(
            (int(target_height / density) + 1, int(target_width / density) + 1))  # saliency scores S(t,i,j)

        for i in range(0, target_height, density):
            for j in range(0, target_width, density):
                masked_wide_rgp = self.create_masking(wide_rgb, center=[i, j], size=[240, 480], radius=radius)

                masked_wide_rgp = skimage.color.gray2rgb(masked_wide_rgp)
                masked_wide_rgp = img_as_ubyte(masked_wide_rgp)

                _masking_wide_rgp = masked_wide_rgp[self.agent.wide_crop_top:, :, :3]
                _masking_wide_rgp = _masking_wide_rgp[..., ::-1].copy()
                _masking_wide_rgp = torch.tensor(_masking_wide_rgp[None]).float().permute(0, 3, 1, 2).to(
                    self.agent.device)

                steer_logits, throt_logits, brake_logits = self.agent.image_model.policy(_masking_wide_rgp, None,
                                                                                         cmd_value)

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
        # res_throttle = pmax_throttle * scores_throttle / scores_throttle.max()

        pmax_brake = scores_brake.max()
        scores_brake = imresize(scores_brake, size=[240, 480], interp='lanczos').astype(np.float32)
        # res_brake = pmax_brake * scores_brake / scores_brake.max()

        pmax_steer = scores_steer.max()
        scores_steer = imresize(scores_steer, size=[240, 480], interp='lanczos').astype(np.float32)
        # res_steer = pmax_steer * scores_steer / scores_steer.max()

        # res = res/100.

        log = get_time_mils()
        tz = pytz.timezone('Europe/Berlin')
        time_stamp = str(datetime.now(tz))
        score_img_name_throttle = f'experiments/scores_throttle_{log}_{time_stamp}_lanczos.png'
        score_img_name_brake = f'experiments/scores_brake_{log}_{time_stamp}_lanczos.png'
        score_img_name_steer = f'experiments/scores_steer_{log}_{time_stamp}_lanczos.png'
        # res_img_name_throttle = f'experiments/res_{log}_{time_stamp}_lanczos.png'
        cv2.imwrite(score_img_name_throttle, scores_throttle)
        cv2.imwrite(score_img_name_brake, scores_brake)
        cv2.imwrite(score_img_name_steer, scores_steer)
        # cv2.imwrite(res_img_name_throttle, res_throttle)
        score_image_throttle = cv2.imread(score_img_name_throttle)
        score_image_brake = cv2.imread(score_img_name_brake)
        score_image_steer = cv2.imread(score_img_name_steer)

        os.remove(score_img_name_throttle)
        os.remove(score_img_name_brake)
        os.remove(score_img_name_steer)

        scores_denoised_throttle = cv2.fastNlMeansDenoising(score_image_throttle, None, 10, 7, 21)
        scores_denoised_brake = cv2.fastNlMeansDenoising(score_image_brake, None, 10, 7, 21)
        scores_denoised_steer = cv2.fastNlMeansDenoising(score_image_steer, None, 10, 7, 21)
        movsd_throttle = np.argmax(np.bincount(scores_denoised_throttle.flat))
        movsd_brake = np.argmax(np.bincount(scores_denoised_brake.flat))
        movsd_steer = np.argmax(np.bincount(scores_denoised_steer.flat))
        erased_gray_score_throttle = np.where(scores_denoised_throttle <= movsd_throttle + 55, 0,
                                              scores_denoised_throttle)
        erased_gray_score_throttle = skimage.color.rgb2gray(erased_gray_score_throttle)
        new_res_throttle = pmax_throttle * erased_gray_score_throttle / erased_gray_score_throttle.max()

        erased_gray_score_brake = np.where(scores_denoised_brake <= movsd_brake + 55, 0,
                                           scores_denoised_brake)
        erased_gray_score_brake = skimage.color.rgb2gray(erased_gray_score_brake)
        new_res_brake = pmax_brake * erased_gray_score_brake / erased_gray_score_brake.max()

        erased_gray_score_steer = np.where(scores_denoised_steer <= movsd_steer + 55, 0,
                                           scores_denoised_steer)
        erased_gray_score_steer = skimage.color.rgb2gray(erased_gray_score_steer)
        # FIXME erased_gray_score_steer.max() = 0?
        # logger.info(f'pmax_steer={pmax_steer}, erased_gray_score_steer={erased_gray_score_steer}, erased_gray_score_steer.max()={erased_gray_score_steer.max()}')
        new_res_steer = pmax_steer * erased_gray_score_steer / erased_gray_score_steer.max()
        # cv2.imwrite(f'experiments/scores_denoised_{log}_{time_stamp}_lanczos.png', scores_denoised_throttle)
        # cv2.imwrite(f'experiments/scores_denoised_outgrayed_larger_scale_{log}_{time_stamp}_lanczos.png', erased_gray_score_throttle)
        # cv2.imwrite(f'experiments/res_denoised_outgrayed_larger_scale{log}_{time_stamp}_lanczos.png', new_res_throttle)
        return [new_res_throttle, new_res_brake, new_res_steer]

    def apply_saliency(self, saliency, frame, fudge_factor=400, channel=0, sigma=0):
        # sometimes saliency maps are a bit clearer if you blur them
        # slightly...sigma adjusts the radius of that blur
        pmax = saliency.max()
        s = imresize(saliency, size=[240, 480], interp='lanczos').astype(np.float32)  # Double it like in origian;
        s = s if sigma == 0 else gaussian_filter(s, sigma=sigma)
        s -= s.min()
        s = fudge_factor * pmax * s / s.max()
        image = frame.astype('uint16')
        image[:, :, channel] += s.astype('uint16')
        image = image.clip(1, 255).astype('uint8')
        return image

    def get_mask(self, center, size, radius):
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        mask = gaussian_filter(mask, sigma=radius)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
        return mask / mask.max()

    def create_masking(self, wide_rgb, center, size, radius):
        occlude = lambda image, mask: image * (1 - mask) + gaussian_filter(image, sigma=3) * mask
        mask = self.get_mask(center, size, radius)
        im = occlude(wide_rgb.squeeze(), mask).reshape(240, 480)
        return im
