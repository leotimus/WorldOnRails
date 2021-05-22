import io
import os

import flask
import torch
from flask import request, make_response

from autoagents.image_agent import ImageAgent
from explainer.explainer import Explainer
from explainer.utils import logger

app = flask.Flask(__name__)
app.config["DEBUG"] = True

image_agent = ImageAgent('saved_model/nocrash/config_nocrash.yaml')
explainer = Explainer(image_agent, [], [])

if os.cpu_count() >= 8:
    num_threads = (os.cpu_count() - 2) / 2
    logger.info(f'Setting pytorch to use {num_threads} cpus.')
    torch.set_num_threads(int(num_threads))


@app.route('/agent/policy', methods=['POST'])
def policy():
    data = request.get_data()
    index = request.headers.get('index')
    buffer = io.BytesIO(data)
    frame = torch.load(buffer)
    throttle, brake, steer = explainer.create_and_save_saliency_ffmpeg(frame, int(index))
    saliency = torch.tensor([throttle, brake, steer])
    buffer = io.BytesIO()
    torch.save(saliency, buffer)
    response = make_response(buffer.getvalue())
    response.headers['content-type'] = 'application/octet-stream'
    return response


app.run(host='0.0.0.0')
