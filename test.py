
# Try and use the VIP transformer in pytorchrl

from torchrl.envs.transforms import VIPTransform, Compose, ToTensorImage
from torchvision.transforms import Normalize
import numpy as np
from tensordict import TensorDict
import torch

def main():

    imageTrans = ToTensorImage(in_keys=["pixels"])
    image = torch.randint(0, 255, (128, 128, 3), dtype=torch.uint8)
    print(image)
    td = TensorDict({"pixels": image})
    # _ = imageTrans(td)
    # print(td)
    # print(td["pixels"])

    transform = VIPTransform("resnet50", size=128)

    _= transform(td)

    print(td)
    print(td["vip_vec"])
    print(td["vip_vec"].shape)
    print("done")

    doMultiple()

def doMultiple():
    transform = VIPTransform("resnet50", size=128)
    mins = []
    maxs = []
    for i in range(1000):
        image = torch.randint(0, 255, (128, 128, 3), dtype=torch.uint8)
        td = TensorDict({"pixels": image})
        _ = transform(td)
        max_val = torch.max(td["vip_vec"])
        min_val = torch.min(td["vip_vec"])
        mins.append(min_val)
        maxs.append(max_val)
        print("Min is " + str(min_val) + " and max is " + str(max_val))
    
    print("Average min is " + str(sum(mins) / len(mins)) + " and average max is " + str(sum(maxs) / len(maxs)))
    print("Min is " + str(min(mins)) + " and max is " + str(max(maxs)))

if __name__ == "__main__":
    main()
