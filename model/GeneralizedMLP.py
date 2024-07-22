import numpy as np
import scipy
import scipy.special as sc
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def calculate_hidden_neurons(input_size, output_size):
    rule_1 = max(input_size, output_size)
    rule_2 = int(2/3 * input_size + output_size)
    rule_3 = min(2 * input_size - 1, input_size + input_size // 3)
    hidden_neurons = min(rule_1, rule_2, rule_3)

    print('hidden neurons: ', hidden_neurons)
    return hidden_neurons

class GeneralizedMLP(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, num_neurons = -1):
        super(GeneralizedMLP, self).__init__()

        if(num_neurons == -1):
            hidden_neurons = calculate_hidden_neurons(input_size, output_size)
        else:
            hidden_neurons = num_neurons
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_neurons))
        
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
        
        self.output_layer = nn.Linear(hidden_neurons, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.sigmoid(layer(x))
        
        x = self.output_layer(x)
        return x
    
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20):
    running_loss_array = []
    val_loss_array = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        running_loss_array.append(running_loss/len(train_loader))
        val_loss_array.append(val_loss/len(val_loader))
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')
    
    return running_loss_array, val_loss_array

def evaluate_model(model, criterion, test_loader):
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    print(f'Test Loss: {test_loss/len(test_loader)}')

    return outputs