import torch
from torch import nn
import torch.nn.functional as F
import time

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        #create hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
    
    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x)) #add relu to each hidden node
            x = self.dropout(x) #add dropout
        x = self.output(x) #add output weights
        return F.log_softmax(x, dim=1) # activation log softmax
    
#train the network
def train_model(model, train_loader, valid_loader, epochs, print_every, criterion, optimizer, scheduler, gpu):
    
    since = time.time()
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"
    
    steps=0
    #to cuda
    model.to(device)
    model.train() 

    for e in range(epochs):
        scheduler.step() 
        running_loss=0
        for ii, (inputs,labels) in enumerate(train_loader):
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device) #gpu
            optimizer.zero_grad() #zero out the gradients 
            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            
            if steps % print_every == 0:
                accuracy,valid_loss = accuracy_loss(model, valid_loader,criterion,gpu)
                
                #print data for the user
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valid_loss),
                      "Validation Accuracy: {:.4f}".format(accuracy))
                
                running_loss = 0
            
            model.train()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    
def accuracy_loss(model, loader, criterion, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"

    model.eval()
    accuracy = 0
    loss=0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            prob = torch.exp(outputs) 
            result = (labels.data == prob.max(1)[1]) 
           
       
            accuracy+=result.type_as(torch.FloatTensor()).mean() 
            loss+=criterion(outputs,labels)     
    return accuracy/len(loader), loss/len(loader) 
