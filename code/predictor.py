import clip
import json
import logging
import logging.config
import numpy as np
import os
import requests
import shutil
import torch
import uuid

from flask import Flask, request, Response, jsonify
from functools import lru_cache
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from urllib.parse import urlparse

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# gunicorn_error_logger = logging.getLogger('gunicorn.error')
# app.logger.handlers.extend(gunicorn_error_logger.handlers)
# app.logger.setLevel(logging.DEBUG)

PREFIX_PATH = "/opt/ml/"
IMAGES_FOLDER = os.path.join(PREFIX_PATH, "images")


def download_images(request_id, image_urls):
    def download_file_from_url(folder, url):
        filename = os.path.join(folder, os.path.basename(urlparse(url).path))
        try:
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)

            return filename
        except Exception:
            return None

    logging.info(f"Downloading {len(image_urls)} images...")

    folder = os.path.join(IMAGES_FOLDER, request_id)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    files = []
    for url in image_urls:
        fragments = urlparse(url, allow_fragments=False)
        if fragments.scheme in ("http", "https"):
            filename = download_file_from_url(folder, url)
        else:
            filename = url

        if filename is None:
            raise Exception(f"There was an error downloading image {url}")

        files.append(filename)

    return files


def delete_images(request_id):
    directory = os.path.join(IMAGES_FOLDER, request_id)

    try:
        shutil.rmtree(directory)
    except OSError as e:
        logging.error(f"Error deleting image directory {directory}.")


class Predictor(object):
    clip_model = None
    device = None
    preprocess = None
    image_mean = None
    image_std = None

    @staticmethod
    def load():
        if Predictor.clip_model is None:
            Predictor.device = "cuda" if torch.cuda.is_available() else "cpu"
            Predictor.clip_model, _ = clip.load("ViT-B/32", device=Predictor.device)

        if Predictor.preprocess is None:
            input_resolution = Predictor.clip_model.input_resolution.item()
            context_length = Predictor.clip_model.context_length.item()
            vocab_size = Predictor.clip_model.vocab_size.item()

            logging.info(
                f"Model parameters: {np.sum([int(np.prod(p.shape)) for p in Predictor.clip_model.parameters()]):,}",
            )
            logging.info(f"Input resolution: {input_resolution}")
            logging.info(f"Context length: {context_length}")
            logging.info(f"Vocab size: {vocab_size}")

            Predictor.preprocess = Compose(
                [
                    Resize(input_resolution, interpolation=Image.BICUBIC),
                    CenterCrop(input_resolution),
                    ToTensor(),
                ]
            )

            Predictor.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(
                Predictor.device
            )
            Predictor.image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(
                Predictor.device
            )

        return Predictor.clip_model

    @staticmethod
    def predict(files, classes):
        images = []

        for f in files:
            image = Predictor.preprocess(Image.open(f).convert("RGB"))
            images.append(image)

        image_input = torch.tensor(np.stack(images)).to(Predictor.device)
        image_input -= Predictor.image_mean[:, None, None]
        image_input /= Predictor.image_std[:, None, None]

        text_input = torch.cat(
            [clip.tokenize(f"a photo of a {c}") for c in classes]
        ).to(Predictor.device)

        with torch.no_grad():
            image_features = (
                Predictor.clip_model.encode_image(image_input)
                .float()
                .to(Predictor.device)
            )

            text_features = (
                Predictor.clip_model.encode_text(text_input)
                .float()
                .to(Predictor.device)
            )

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        result_count = min(len(classes), 3)
        values, indices = similarity.cpu().topk(result_count, dim=-1)

        response = []

        for index, value in zip(indices, values):
            indices = index.numpy().tolist()
            response.append(
                (
                    indices,
                    list(map(lambda i: classes[i], indices)),
                    value.numpy().tolist(),
                )
            )

        return response


Predictor.load()


@app.route("/ping", methods=["GET"])
def ping():
    """This endpoint determines whether the container is working and healthy."""
    logging.info("Ping received...")

    health = Predictor.load() is not None

    status = 200 if health else 404
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invoke():
    if request.content_type != "application/json":
        return Response(
            response='{"reason" : "Request should be application/json"}',
            status=400,
            mimetype="application/json",
        )

    data = request.get_json()

    request_id = uuid.uuid4().hex
    files = download_images(request_id=request_id, image_urls=data["images"])

    Predictor.load()
    result = Predictor.predict(files=files, classes=data["classes"])

    delete_images(request_id=request_id)

    return Response(
        response=json.dumps(result),
        status=200,
        mimetype="application/json",
    )
