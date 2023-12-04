import base64
import open_clip
import cv2
import io
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


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# gunicorn_error_logger = logging.getLogger('gunicorn.error')
# app.logger.handlers.extend(gunicorn_error_logger.handlers)
# app.logger.setLevel(logging.DEBUG)

PREFIX_PATH = "/opt/ml/"
IMAGES_FOLDER = os.path.join(PREFIX_PATH, "images")


def download_image(request_id, image):
    def download_file_from_url(folder, url):
        filename = os.path.join(folder, os.path.basename(urlparse(url).path))
        try:
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)

            return filename
        except Exception:
            return None

    logging.info(f'Downloading image "{image}"...')

    folder = os.path.join(IMAGES_FOLDER, request_id)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    fragments = urlparse(image, allow_fragments=False)
    if fragments.scheme in ("http", "https"):
        filename = download_file_from_url(folder, image)
    else:
        filename = image

    if filename is None:
        raise Exception(f"There was an error downloading image {image}")

    return Image.open(filename).convert("RGB")


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
    tokenizer = None

    @staticmethod
    def load():
        print('MASSIVE PENIS')
        if Predictor.clip_model is None:
            Predictor.device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            model.eval()
            Predictor.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            Predictor.clip_model = model.to(Predictor.device)

            Predictor.preprocess = preprocess

        return Predictor.clip_model

    @staticmethod
    def embed_images(images):
        images = [Predictor.preprocess(image) for image in images]
        image_input = torch.tensor(np.stack(images)).to(Predictor.device)

        with torch.no_grad():
            return (
                Predictor.clip_model.encode_image(image_input)
                .float()
                .to(Predictor.device)
            )

    @staticmethod
    def embed_text(text):
        text_input = Predictor.tokenizer(text).to(Predictor.device)

        with torch.no_grad():
            return (
                Predictor.clip_model.encode_text(text_input)
                .float()
                .to(Predictor.device)
            )

    @staticmethod
    def predict(images, classes):
        image_features = Predictor.embed_images(images)

        text_features = Predictor.embed_text(
            [f"a photo of a {c}" for c in classes]
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


@app.route("/embeddings", methods=["POST"])
def embeddings():
    if request.content_type != "application/json":
        return Response(
            response='{"reason" : "Request should be application/json"}',
            status=400,
            mimetype="application/json",
        )

    Predictor.load()

    request_id = uuid.uuid4().hex

    data = request.get_json()

    images = []
    for im in data.get("images", []):
        fragments = urlparse(im, allow_fragments=False)
        if fragments.scheme in ("http", "https", "file"):
            image = download_image(request_id, im)
        else:
            image = Image.open(io.BytesIO(base64.b64decode(im)))

        images.append(image)

    result = {
        'images': Predictor.embed_images(images).tolist() if len(images) > 0 else [],
        'texts': Predictor.embed_text(data.get("texts", [])).tolist()
    }

    delete_images(request_id=request_id)

    return Response(
        response=json.dumps(result),
        status=200,
        mimetype="application/json",
    )

@app.route("/invocations", methods=["POST"])
def invoke():
    if request.content_type != "application/json":
        return Response(
            response='{"reason" : "Request should be application/json"}',
            status=400,
            mimetype="application/json",
        )

    Predictor.load()

    request_id = uuid.uuid4().hex

    data = request.get_json()

    images = []
    for im in data.get("images", []):
        fragments = urlparse(im, allow_fragments=False)
        if fragments.scheme in ("http", "https", "file"):
            image = download_image(request_id, im)
        else:
            image = Image.open(io.BytesIO(base64.b64decode(im)))

        images.append(image)

    result = Predictor.predict(images=images, classes=data["classes"])

    delete_images(request_id=request_id)

    return Response(
        response=json.dumps(result),
        status=200,
        mimetype="application/json",
    )
