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
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store', help='directory containing images')
parser.add_argument('--save_dir', action='store', help='save trained checkpoint to this directory' )
parser.add_argument('--arch', action='store', help='what kind of pretrained architecture to use', default='vgg19')
parser.add_argument('--gpu', action='store_true', help='use gpu to train model')
parser.add_argument('--epochs', action='store', help='# of epochs to train', type=int, default=9)
parser.add_argument('--learning_rate', action='store', help='which learning rate to start with', type=float, default=0.1)
parser.add_argument('--hidden_units', action='store', help='# of hidden units to add to model', type=int, default=4096)
parser.add_argument('--output_size', action='store', help='# of classes to output', type=int, default=102)

args=parser.parse_args()

#sort data for training, validation, testing
data_dir = args.data_dir

train_data, valid_data, test_data, train_loader, valid_loader, test_loader = utility_functions.preprocess_image(data_dir)

#create model w/ vgg19 as default
model = utility_functions.torch_model(args.arch)

#freeze paremeters
for param in model.parameters():
    param.requires_grad = False#params are now frozen so that we do not backprop thru them again

#calculate input size into the network classifier
input_size = utility_functions.get_input_size(model, args.arch)

model.classifier = model_functions.Model(input_size, args.output_size, [args.hidden_units], drop_p=0.5)

#define criterion, optimizer, and scheduler
criterion = nn.NLLLoss() # nllloss b/c the logsoftmax is our output activation
optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

#train the model!
model_functions.train_model(model, train_loader, valid_loader, args.epochs, 32, criterion, optimizer, scheduler, args.gpu)

#test the model!
test_accuracy, test_loss = model_functions.accuracy_loss(model, test_loader, criterion, args.gpu)
print("\n ---\n Test Accuracy: {:.2f} %".format(test_accuracy*100), "Test Loss: {}".format(test_loss))

#save the network to the checkpoint
utility_functions.save_model(model, train_data, optimizer, args.save_dir, args.arch)