import json

import torch

from autoagents.image_agent import ImageAgent
# split input into smaller sublist, one input might contain 1001 entries
from explainer.explainer import Explainer
from explainer.utils import logger


def split_data(start_index, n):
    Ls = torch.load('experiments/flush_1620568583468.data')
    data = []
    for i in range(n):
        data.append(Ls[start_index + i])
    torch.save(data, f'experiments/flush_1620568574409_{start_index}_{n}.data')


def run_all_ffmpeg(config, data):
    logger.info(f'Initializing rails CameraModel with config: {config}')
    image_agent = ImageAgent(config)
    data = torch.load(data)
    explainer = Explainer(image_agent, data, [])
    explainer.explain()


def run_all_ffmpeg_remote(config, data, hosts):
    logger.info(f'Initializing rails CameraModel with config: {config}, host list {hosts}')
    image_agent = ImageAgent(config)
    data = torch.load(data)
    explainer = Explainer(image_agent, data, hosts)
    explainer.explain()


def run_analzyse_data_set(config, data):
    image_agent = ImageAgent(config)
    data = torch.load(data)
    explainer = Explainer(image_agent, data)
    explainer.analzye_data(data)


def read_hosts(config='experiments/hosts.json') -> list:
    with open(config) as config_file:
        return json.load(config_file)['hosts']


if __name__ == '__main__':
    # split_data(800, 120)
    # run_all('saved_model/nocrash/config_nocrash.yaml', 'experiments/flush_1620568574409_970_1.data')
    # run_analzyse_data_set('saved_model/nocrash/config_nocrash.yaml', 'experiments/flush_1620568574409.data')
    # run_all_ffmpeg('saved_model/nocrash/config_nocrash.yaml', 'experiments/data_01/flush_1620568574409_200_2.data')
    run_all_ffmpeg_remote('saved_model/nocrash/config_nocrash.yaml',
                          'experiments/flush_1620568578861.data',
                          read_hosts())
