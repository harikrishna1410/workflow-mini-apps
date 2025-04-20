import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from .component import Component
import time

class SimpleFeedForwardNet(nn.Module):
    def __init__(self, dropout=0.1, use_batchnorm=True, num_layers = 0):
        super(SimpleFeedForwardNet, self).__init__()
        
        self.model = nn.Sequential()
        self.neurons_per_layer = 256
        self.nlayers = num_layers
        neurons_per_layer = self.neurons_per_layer
        self.layer = [nn.Linear(neurons_per_layer, neurons_per_layer),nn.ReLU()]
        if use_batchnorm:
            self.layer.append(nn.BatchNorm1d(neurons_per_layer))
        if dropout > 0:
            self.layer.append(nn.Dropout(dropout))

        for i in range(num_layers):
            self.model.add_module(f"{i}", nn.Sequential(*self.layer))
    
    def add_layer(self):
        """Add a new layer to the model."""
        self.model.add_module(str(len(self.model)), nn.Sequential(*self.layer))
        self.nlayers += 1
    
    def remove_layer(self, layer_index=-1):
        """Remove a layer from the model."""
        if layer_index < len(self.model):
            del self.model[layer_index]

    def forward(self, x):
        return self.model(x)
        
def setup_dataloader(data_set_size, x_shape, y_shape, batch_size=32, shuffle=True, ddp=False):
    X = torch.randn((data_set_size, *x_shape), dtype=torch.float32)
    y = torch.randn((data_set_size, *y_shape), dtype=torch.float32)
    dataset = TensorDataset(X, y)
    if ddp:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def create_loss_function(loss_type="mse"):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloader.dataset)


class AI(Component):
    def __init__(self,
                 name = "AI", 
                model_type="feedforward", 
                dropout=0.1, 
                use_batchnorm=True, 
                num_layers=0, 
                loss_type="mse", 
                lr=0.001, 
                data_size=1000, 
                batch_size=32, 
                shuffle=True, 
                ddp=False, 
                num_epochs=10):
        super().__init__(name)
        self.name = name
        self.model_type = model_type
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.num_layers = num_layers
        self.loss_type = loss_type
        self.lr = lr
        self.data_size = data_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ddp = ddp
        self.num_epochs = num_epochs
        

        self.model = self.build_model()
        self.device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = None
        self.optimizer = None
            
    def build_model(self):
        """Build a neural network model."""
        if self.model_type == "feedforward":
            self.model = SimpleFeedForwardNet(dropout=self.dropout, 
                                              use_batchnorm=self.use_batchnorm, 
                                              num_layers=self.num_layers)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        return self.model
        
    def setup_training(self):
        """Set up loss function and optimizer for training."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
            
        self.criterion = create_loss_function(self.loss_type)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
    def train(self):
        """Train the model using data specifications to build a dataloader."""
        if self.model is None or self.criterion is None or self.optimizer is None:
            if self.model is None:
                self.build_model()
            self.setup_training()

        dataloader = setup_dataloader(  self.data_size, 
                                        (self.model.neurons_per_layer,), 
                                        (self.model.neurons_per_layer,), 
                                        self.batch_size, 
                                        self.shuffle, 
                                        self.ddp)
        train(self.model, dataloader, self.criterion, self.optimizer, self.device, self.num_epochs)
        
    def set_nlayers_train(self,total_time):
        """
        Determine the number of layers based on the total time.
        """
        simulated_time = 0
        while simulated_time < total_time:
            self.model.add_layer()
            self.setup_training()
            tic = time.perf_counter()
            self.train()
            toc = time.perf_counter()
            simulated_time = toc - tic
        return 

    def infer(self):
        """Perform inference on inputs."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        inputs = torch.randn((self.batch_size, self.model.neurons_per_layer), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
        return outputs