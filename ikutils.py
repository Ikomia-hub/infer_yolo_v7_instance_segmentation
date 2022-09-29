import requests
import os

model_zoo = {'yolov7-seg': "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt"}


def download_model(name, models_folder):
    URL = model_zoo[name]
    print("Downloading model for {}".format(name))
    response = requests.get(URL)
    with open(os.path.join(models_folder, name + ".pt"), "wb") as f:
        f.write(response.content)


def clamp(x, mini, maxi):
    if x > maxi:
        return maxi
    if x < mini:
        return mini
    return x
