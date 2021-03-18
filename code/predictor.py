import clip
import ftfy
import gzip
import html
import json
import logging
import logging.config
import numpy as np
import os
import regex as re
import requests
import shutil
import torch
import uuid

from flask import Flask, request, Response, jsonify
from functools import lru_cache
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from urllib.parse import urlparse


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# gunicorn_error_logger = logging.getLogger('gunicorn.error')
# app.logger.handlers.extend(gunicorn_error_logger.handlers)
# app.logger.setLevel(logging.DEBUG)

PREFIX_PATH = "/opt/ml/"
IMAGES_FOLDER = os.path.join(PREFIX_PATH, "images")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text


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

        return Predictor.clip_model

    def initialize():
        if Predictor.preprocess is not None:
            return

        if Predictor.clip_model is None:
            Predictor.load()
            return

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

    @staticmethod
    def predict(files, image_urls, classes):
        images = []

        for file in files:
            image = Predictor.preprocess(Image.open(file).convert("RGB"))
            images.append(image)

        image_input = torch.tensor(np.stack(images)).to(Predictor.device)
        image_input -= Predictor.image_mean[:, None, None]
        image_input /= Predictor.image_std[:, None, None]

        tokenizer = SimpleTokenizer(
            bpe_path=os.path.join(PREFIX_PATH, "bpe_simple_vocab_16e6.txt.gz")
        )

        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]

        text_descriptions = [f"This is a photo of a {label}" for label in classes]
        text_tokens = [
            [sot_token] + tokenizer.encode(desc) + [eot_token]
            for desc in text_descriptions
        ]
        text_input = torch.zeros(
            len(text_tokens), Predictor.clip_model.context_length, dtype=torch.long
        )

        for i, tokens in enumerate(text_tokens):
            text_input[i, : len(tokens)] = torch.tensor(tokens)

        text_input = text_input.to(Predictor.device)

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
            text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        result_count = min(len(classes), 5)
        top_probs, top_labels = text_probs.cpu().topk(result_count, dim=-1)

        response = []

        for i, file in enumerate(files):
            response.append(
                {
                    "url": image_urls[i],
                    "labels": [classes[label] for label in top_labels[i].numpy()],
                    "probs": [float(prob) for prob in top_probs[i].numpy()],
                }
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

    Predictor.initialize()

    data = request.get_json()

    request_id = uuid.uuid4().hex
    files = download_images(request_id=request_id, image_urls=data["images"])

    Predictor.load()
    result = Predictor.predict(
        files=files, image_urls=data["images"], classes=data["classes"]
    )

    delete_images(request_id=request_id)

    return Response(
        response=json.dumps(result),
        status=200,
        mimetype="application/json",
    )
