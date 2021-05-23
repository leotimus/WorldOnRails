import io
import os

import flask
import numpy
import torch
from flask import request, make_response

from autoagents.image_agent import ImageAgent
from explainer.explainer import Explainer
from explainer.utils import logger

app = flask.Flask(__name__)
app.config["DEBUG"] = True

image_agent = ImageAgent('saved_model/nocrash/config_nocrash.yaml')
explainer = Explainer(image_agent, [], [])

cpu_count = os.cpu_count()
if cpu_count >= 8:
    num_threads = (cpu_count - 2) / 2
    logger.info(f'Setting pytorch to use {num_threads} cpus per process.')
    torch.set_num_threads(int(num_threads))


@app.route('/agent/policy', methods=['POST'])
def policy():
    data = request.get_data()
    index = request.headers.get('index')
    buffer = io.BytesIO(data)
    frame = torch.load(buffer)
    try:
        throttle, brake, steer = explainer.create_and_save_saliency_ffmpeg(frame, int(index))
    except Exception as error:
        logger.error(f'Got exception, return empty result', error)
        throttle = numpy.empty(shape=[4, 3])
        brake = numpy.empty(shape=[4])
        steer = numpy.empty(shape=[4, 9])
    saliency = torch.tensor([throttle, brake, steer])
    buffer = io.BytesIO()
    torch.save(saliency, buffer)
    response = make_response(buffer.getvalue())
    response.headers['content-type'] = 'application/octet-stream'
    return response


app.run(host='0.0.0.0')
