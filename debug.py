from autoagents.image_agent import ImageAgent
import torch

# split input into smaller sublist, one input might contain 1001 entries
def split_data(start_index, n):
    Ls = torch.load('experiments/flush_1620568574409.data')
    data = []
    for i in range(n):
        data.append(Ls[start_index + i])
    torch.save(data, f'experiments/flush_1620568574409_{start_index}_{n}.data')


def run_all(config, data):
    imageAgent = ImageAgent(config)
    Ls = torch.load(data)
    imageAgent.saveSaliencyVideo(Ls)


if __name__ == '__main__':
    # split_data(200, 60)
    run_all('saved_model/nocrash/config_nocrash.yaml', 'experiments/flush_1620568574409_200_60.data')
