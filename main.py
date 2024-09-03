import argparse
import os
import tarfile

import requests
import tensorflow_hub as hub
import websockets.sync.client as websocket
from tqdm import tqdm

import worker


def download_model(model):
    response = requests.get(f"https://tfhub.dev/{model}?tf-hub-format=compressed", stream=True)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024

    with open("model.tar.gz", "wb") as file:
        for data in tqdm(response.iter_content(chunk_size=chunk_size),
                         total=total_size // chunk_size,
                         unit="MB",
                         unit_scale=True,
                         ascii=True,
                         desc="Downloading model"):
            file.write(data)

    with tarfile.open("model.tar.gz", "r:gz") as file:
        file.extractall("model")
    os.remove("model.tar.gz")
    print(f"Downloaded model {model}")

    return


def main():
    parser = argparse.ArgumentParser(description='AIC Swarm Manager')
    parser.add_argument('--url',
                        type=str,
                        default='localhost:8000',
                        help='The URL to connect to')
    parser.add_argument("--model",
                        type=str,
                        default="google/faster_rcnn/openimages_v4/inception_resnet_v2/1",
                        help="The model to use")

    args = parser.parse_args()
    url = args.url
    model = args.model
    ws_url = f"ws://{url}/session"

    if not url or not model:
        raise ValueError("URL and model must be provided")

    if not os.path.exists("model"):
        download_model(model)

    model = hub.load("model")
    detector = model.signatures["default"]
    print(f"Loaded model {model}")

    processed = 0
    with websocket.connect(ws_url) as ws:
        print("Connected to server")
        while True:
            try:
                worker.process(ws, args.url, detector)
                processed += 1
            except Exception as error:
                print(f"Processed {processed} items, good job :D")
                print("Find error, raising....")
                raise error


if __name__ == "__main__":
    main()
