import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import time
import h5py
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

class SRCNN(nn.Module):
  def __init__(self):
    super(SRCNN, self).__init__()

    self.conv1 = nn.Conv2d(1, 64, kernel_size = 9, padding = 2, padding_mode = 'replicate')
    self.conv2 = nn.Conv2d(64, 32, kernel_size = 1, padding = 2, padding_mode = 'replicate')
    self.conv3 = nn.Conv2d(32, 1, kernel_size = 5, padding = 2, padding_mode = 'replicate')

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.conv3(x)

    return x

model = SRCNN().to(device)

def load_checkpoint(filepath):
  model = SRCNN()
  model.load_state_dict(torch.load(filepath, map_location = device))
  model.eval()

  return model


def resize(img, new_size = (96, 96)):
  img = img.resize(new_size)
  return img


def get_prediction(img_path):
  img = Image.open(img_path).convert('YCbCr')
  original_img_size = img.size
  img = resize(img)
  img = img.resize((int(img.width * 3), int(img.height * 3)), Image.BICUBIC)

  y, cb, cr = img.split()
  img_to_tensor = transforms.ToTensor()
  img = img_to_tensor(y).view(1, -1, y.size[1], y.size[0]).to(device)
  
  model = load_checkpoint("./train_res_400_epochs.pth")
  out_srcnn = model(img)

  out_img = out_srcnn[0].cpu().detach().numpy()
  out_img *= 255.0
  out_img = out_img.clip(0, 255)
  out_img = Image.fromarray(np.uint8(out_img[0]), mode = 'L')

  out_img_final = Image.merge('YCbCr', [out_img, cb, cr]).convert('RGB')
  
  return out_img_final

