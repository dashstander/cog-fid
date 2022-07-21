# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, Input, Path

from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.resize import make_resizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
from replicate.client import Client
import requests
from scipy.stats import multivariate_normal
from torchvision.transforms import ToTensor, Compose
import tempfile
from typing import List, Tuple



def make_image(prompt, model_name, api_token, seed):
    model = Client(api_token).models.get(model_name)
    image_url = list(model.predict(prompt=prompt, seed=seed))[-1]
    return model_name, Image.open(requests.get(image_url, stream=True).raw)


def make_output(pairs: List[Tuple[str, float, PIL.Image]]) -> Path:
    rows = len(pairs)
    columns = 1
    fig = plt.figure(figsize=(10, 7))
    for i, (model, nll, image) in enumerate(pairs):
        info = f'{model}, negative log-likelihood: {nll:.3f}'
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(info)
    output_path = Path(tempfile.mkdtemp()) / "images.png"
    fig.savefig(output_path)
    return output_path



class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = build_feature_extractor('clean')
        self.ref_mu, self.ref_cov = get_reference_statistics('cifar10', 32)
        resize_fn =  make_resizer('PIL', False, 'bicubic', (32, 32))
        self.image_fn = Compose([np.asarray, resize_fn, ToTensor()])

    def get_fid_features(self, image):
        return self.model(
            self.image_fn(image).to('cuda:0')
        )

    def predict(
        self,
        prompt: str = Input(description='The prompt to test on different models.'),
        model_urls: str = Input(description='Replicate model paths, pipe delimited'),
        api_token: str = Input(description='Your API token'),
        seed: int = Input(description='Seed for reproducibility')
    ) -> Path:
        """Run a single prediction on the model"""
        urls = [url.strip() for url in model_urls.split('||')]
        futures = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for url in urls:
                args = (prompt, url, api_token, seed)
                futures.append(
                    executor.submit(make_image, *args)
                )
        outs = []
        for fut in as_completed(futures):
            model_name, image = fut.result()
            features = self.get_fid_features(image)
            likelihood = multivariate_normal.logpdf(features, self.ref_mu, self.ref_cov)
            outs.append(
                (model_name, -1. * np.log(likelihood), image)
            )
        return make_output(outs)




        
