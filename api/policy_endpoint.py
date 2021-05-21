import io

import flask
import torch
from flask import request, make_response

from autoagents.image_agent import ImageAgent
from explainer.explainer import Explainer

app = flask.Flask(__name__)
app.config["DEBUG"] = True

image_agent = ImageAgent('saved_model/nocrash/config_nocrash.yaml')
explainer = Explainer(image_agent, [], [])


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


app.run()
