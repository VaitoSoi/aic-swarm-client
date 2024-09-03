import json
import tempfile

import requests
import tensorflow as tf
import websockets.sync.client as websocket
import time
from PIL import Image


def run_detector(img, detector):
    img_tensor = tf.convert_to_tensor(img)
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)[tf.newaxis, ...]
    result = detector(img_tensor)
    result = {key: value.numpy() for key, value in result.items()}
    return result


def parse(result):
    boxes = result["detection_boxes"]
    classes = result["detection_class_entities"]
    scores = result["detection_scores"]

    objects = []
    for i in range(len(boxes)):
        if scores[i] < 0.5:
            continue

        objects.append(
            {
                "box": boxes[i].tolist(),
                "class": classes[i].decode("utf-8"),
                "score": float(scores[i]),
            }
        )

    return objects


def process(ws: websocket.ClientConnection, url: str, detector: str):
    def send(data):
        ws.send(json.dumps(data))

    send(["process", None])
    item = ws.recv()

    with tempfile.SpooledTemporaryFile() as file:
        response = requests.get(f"{url if "http" in url else f"http://{url}"}/file/keyframes/{item}")
        file.write(response.content)

        image = Image.open(file).convert("RGB")
        start = time.time()
        result = run_detector(image, detector)
        print(f"Processed {item} in {(time.time() - start)}s")
        objects = parse(result)

        send(["finish", objects])
