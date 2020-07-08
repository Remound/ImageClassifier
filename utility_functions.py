import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import model_functions
import json

def preprocess_image(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Define transforms for the data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    #datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms) 
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    #dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data)
    test_loader = torch.utils.data.DataLoader(test_data)

    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader

def process_image(image):
    ''' resizes/normalizes a PIL image for a PyTorch model and returns an Numpy array
    '''
    img = Image.open(image)
    img = img.resize((256,256))
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    coordinates = (left,top,right,bottom)
    img = img.crop(coordinates)
    
    #np array
    np_img = np.array(img)
    
    #normalize the image
    np_img = ((np_img / 255)- [0.485, 0.456, 0.406])/([0.229, 0.224, 0.225])
    
    
    np_img = np.ndarray.transpose(np_img, (2,0,1)) 
    img_tensor = torch.from_numpy(np.expand_dims(np_img, axis=0)).float() 

    return img_tensor

#save the model
def save_model(model, train_data, optimizer, save_dir, arch):
    checkpoint = {'input_size': model.classifier.hidden_layers[0].in_features,
                  'output_size': model.classifier.output.out_features,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'state_dict': model.classifier.state_dict(),
                  'class_idx': train_data.class_to_idx,
                  'optim_idx': optimizer.state_dict,
                  'dropout': model.classifier.dropout.p,
                  'arch':arch}
    
    if (save_dir == None): 
        torch.save(checkpoint, 'checkpoint.pth')
    else:
        torch.save(checkpoint, save_dir+'checkpoint.pth')

#load the model from the saved checkpoint
def load_model(filepath, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = "cpu"
    
    checkpoint = torch.load(filepath)
    model = torch_model(checkpoint['arch'])
    new_model = model_functions.Model(checkpoint['input_size'],
                     checkpoint['output_size'],
                     checkpoint['hidden_layers'], 
                     checkpoint['dropout'])
    new_model.load_state_dict(checkpoint['state_dict'])
    model.classifier = new_model
    model = model.float().to(device)
    return model

def torch_model(arch):
     try: 
            model = getattr(models, arch)(pretrained=True)
            return model
     except AttributeError:
         print("%s is not valid torchvision model" % arch)
         raise SystemExit
     else:
        print("error loading model")
        raise SystemExit

        
def get_input_size(model, arch):
    input_size = 0
    
    if('vgg' in arch): return model.classifier[0].in_features
    elif('densenet' in arch): return model.classifier.in_features
    elif('squeezenet' in arch): return model.classifier[1].in_channels
    elif(('resnet' in arch) or ('inception'in arch) ): return model.fc.in_features
    elif('alexnet' in arch): return model.classifier[1].in_features
        
    if(input_size == 0): raise Error    
    return input_size

def get_the_class(classes, checkpoint, category_names):
    class_to_idx = torch.load(checkpoint)['class_idx'] 
    
    idx_to_class = {idx: pic for pic, idx in class_to_idx.items()} 
    
    #change index number to class number and then to flower name
    if(category_names != None): 
        names = []
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        for i in classes:
            category = idx_to_class[i] # index of top5 to class number
            name = cat_to_name[category] #category/class number to flower name
            names.append(name)
        return names
    
    else: 
        class_id = []
        for i in classes:
            class_id.append(idx_to_class[i])
        return class_id

def show_the_classes(probabilities, classes, top_k):
    print('******Predictions for Image******')
    i = 0
    while (i < top_k):
        print('%*s. Class: %*s. Pr= %.4f'%(7, i+1, 3, classes[i], probabilities[i]))
        i += 1

def display_result(image, probabilities, classes, top_k):
    #shows the image
    fig, ax = plt.subplots()
    
    image = np.squeeze(image.numpy(), axis=0).transpose((1, 2, 0))

    # Undo the preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    #show probabilities 
    fig, ax = plt.subplots()
    ax.barh(np.arange(top_k), probabilities)
    ax.set_aspect(0.1)
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(classes, size='small')
    ax.set_title('Class Probability')
    ax.set_xlim(0,max(probabilities)+0.1)
    plt.tight_layout()
    plt.show()
       
    return ax
