import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import loss_ce

def train(model, iter, step_size, gamma, lr):
    device = torch.device('cuda:0')
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               factor=gamma,
                                               patience=step_size)
    
    for image, letter, label in iter['train']:
        image = image.to(device)
        letter = letter.clone().detach().to(device)
        label = torch.tensor([int(x) for x in label]).to(device)
        
        optimizer.zero_grad()
        output = model(image, letter)
        loss = loss_ce(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
            
def train_evaluate(model, iter, data):
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    
    loss = 0.0
    correct = 0.0
    
    for image, letter, label in iter['train']:
        image = image.to(device)
        letter = letter.clone().detach().to(device)
        label = torch.tensor([int(x) for x in label]).to(device)
        
        output = model(image, letter)
        loss = loss_ce(output, label)
        loss += loss.item()*image.size(0)
        loss /= len(data['train'])
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        acc = 100. * correct / len(data['train'])

    print(f'Train Loss: {loss:.4f}, Accuracy: {acc:.2f}%')
    
def eval(model, iter, data):
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    
    loss = 0.0
    correct = 0.0
    
    for image, letter, label in iter['val']:
        image = image.to(device)
        letter = letter.clone().detach().to(device)
        label = torch.tensor([int(x) for x in label]).to(device)
        
        output = model(image, letter)
        loss = loss_ce(output, label)
        loss += loss.item()*image.size(0)
        loss /= len(data['val'])
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        acc = 100. * correct / len(data['val'])
                    
    print(f'Val Loss: {loss:.4f}, Accuracy: {acc:.2f}%')


def train_teacher(epochs, model, iter, data, step_size, gamma, lr):
    print('------Training Teacher------')
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1} / {epochs}')
        train(model, iter, step_size, gamma, lr)
        train_evaluate(model, iter, data)
        eval(model, iter, data)
    return model