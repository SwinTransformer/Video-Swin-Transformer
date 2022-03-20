"""
On the terminal run:
        mkdir weights
        cd weights
        gdown https://drive.google.com/uc?id=10_ArqSj837hBzoQTq3RPGBZgKbBvNfSe
to use the use_pretrained function with default parameters.
"""
import cv2
import requests
import torch
from torchvision import transforms
from tqdm import tqdm

from mmaction.models.recognizers.swintransformer3d import SwinTransformer3D


def video2img(video_path: str):
    """
    Converts a video to a torch tensor of (channels, frames, height, width).
    Args:
        video_path: path to the video.
    Returns:
        torch tensor of (channels, frames, height, width).
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    l = []
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]
        ),
    ])
    while success:
        if count % 20 == 0:
            l.append(
                transform(
                    torch.tensor(image).type(
                        torch.FloatTensor).permute(2, 0, 1)
                ).unsqueeze(dim=0)
            )
        success, image = vidcap.read()
        count += 1
    return torch.stack(l, dim=2)


def use_pretrained(model,
                   folder='weights/',
                   file_name="swint_victim_pretrained.pth",
                   download=False, url=None, ):
    """
    Loads a pretrained model.
    Args:
        model: model to load the weights to.
        folder: folder to load the weights from.
        file_name: name of the file to load the weights from.
        download: whether to download the weights from the url.
        url: url to download the weights from.
    Returns:
        model with loaded weights.
    """
    if download:
        response = requests.get(url, stream=True)
        t = int(response.headers.get('content-length', 0))
        block_size = 1024 ** 2
        progress_bar = tqdm(total=t, unit='iB', unit_scale=True)
        with open(f"weights/{file_name}", 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if (t != 0) and (progress_bar.n != t):
            print("ERROR downloading weights!")
            return -1
        print(f"Weights downloaded in {folder} directory!")
    model.load_state_dict(torch.load(os.path.join(folder, file_name)))
    return model


model = SwinTransformer3D()
use_pretrained(model)
# The input must be of the form (batchSize, channels, frames, height, width)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    student_model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.02
)
