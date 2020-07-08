import model_functions
import utility_functions
import argparse
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument('input', action='store', help='path to image to be classified')
parser.add_argument('--gpu', action='store_true', help='use gpu')
parser.add_argument('checkpoint', action='store', help='path to previously saved model')
parser.add_argument('--category_names', action='store', help='file that maps the classes to the names')
parser.add_argument('--top_k', action='store', type=int, default=3, help='how many class probabilities to show')

args=parser.parse_args()

if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: device = "cpu"

model = utility_functions.load_model(args.checkpoint, args.gpu).eval()

img = utility_functions.process_image(args.input).to(device) 

outputs = model(img) 

probabilities = torch.exp(outputs) 

results = torch.topk(probabilities, args.top_k)    

top_probabilities = results[0][0].cpu().detach().numpy() 

classes = results[1][0].cpu().numpy() 

if(args.category_names != None): 
    classes = utility_functions.get_the_class(classes, args.checkpoint, args.category_names)
else:
    classes=utility_functions.get_the_class(classes, args.checkpoint, None)

#predicted probabilities 
utility_functions.show_the_classes(top_probabilities, classes, args.top_k)