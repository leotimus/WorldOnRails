import io

import numpy
import requests
import torch

from autoagents.image_agent_saliency import ImageAgentSaliency


class RemoteExplainer:

    def __init__(self, host: str):
        self.host = host

    def create_and_save_saliency_ffmpeg(self, info: ImageAgentSaliency, i: int):
        buffer = io.BytesIO()
        torch.save(info, buffer)
        headers = {'index': str(i)}
        res = requests.post(f'{self.host}/agent/policy', data=buffer.getvalue(), headers=headers)
        saliency = torch.load(io.BytesIO(res.content))
        saliency_img_throttle = numpy.array(saliency[0])
        saliency_img_brake = numpy.array(saliency[1])
        saliency_img_steer = numpy.array(saliency[2])
        return saliency_img_throttle, saliency_img_brake, saliency_img_steer
