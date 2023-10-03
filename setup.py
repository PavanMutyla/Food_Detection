! pip install torchinfo
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torchinfo import summary
import os
import zipfile
import requests
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = "cuda" if torch.cuda.is_available() else "cpu"
device
