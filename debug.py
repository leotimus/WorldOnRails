from autoagents.image_agent import ImageAgent
import torch

imageAgent = ImageAgent('/home/demo/Downloads/WorldOnRails/saved_model/nocrash/config_nocrash.yaml')
Ls = torch.load('experiments/flush_1620568574409.data')
imageAgent.saveSaliencyVideo(Ls)