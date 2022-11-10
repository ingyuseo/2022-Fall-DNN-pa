import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, num_features, num_hidden):
        super().__init__()

        self.num_features = num_features
        self.num_hidden = num_hidden
        # ========================= EDIT HERE ==========================
        
        # hidden layer, activation function
        torch.manual_seed(777)
        self.device = 'cpu' 

        # hidden layer, activation function
        self.model = nn.Sequential(
          nn.Linear(num_features, num_hidden, bias=True),
          nn.Sigmoid(),
          nn.Linear(num_hidden, num_hidden, bias=True),
          nn.Sigmoid(),
          nn.Linear(num_hidden, num_hidden, bias=True),
          nn.Sigmoid(),
          nn.Linear(num_hidden, num_hidden, bias=True),
          nn.Sigmoid(),
          nn.Linear(num_hidden, 1, bias=True),
          nn.Sigmoid()
        ).to(self.device)


        # ============================================================

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch
        # ========================= EDIT HERE ==========================

        # Train should be done for 'epochs' times.
        # The function 'train' should return the loss of final epoch.
        # Define  the loss function.


        # start training
        train_size = x.shape[0]
        X = torch.tensor(x)
        Y = torch.tensor(y.reshape(train_size,1)).to(torch.float32)

        for epoch in range(epochs):
            
            loss=0
            indices = torch.randperm(X.size(0))
            X = X[indices]
            Y = Y[indices]

            #2. Train using mini-batch
            minibatches = train_size//batch_size 
            for iter in range(minibatches):
                batch_x = X[iter*batch_size: (iter+1)*batch_size]
                batch_y = Y[iter*batch_size : (iter+1)*batch_size]

                optim.zero_grad()
                hypothesis = self.model(batch_x)

                criterion = nn.BCELoss().to(self.device)
                cost = criterion(hypothesis, batch_y)

                loss += cost.item()
                cost.backward()
                optim.step()
            
            # if epoch % 100 == 0:
            #     print('Epoch {:4d} Cost: {:.6f}'.format(
            #         epoch, cost.item()
            #     ))

        # ============================================================
        return loss

    def forward(self, x):
        y_predicted = None
        # ========================= EDIT HERE ========================
        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # If the activation function is sigmoid,
        # model predicts the label as 0 if the output is smaller than 0.5.
        # Otherwise, it predicts as 1.
        
        X = torch.tensor(x)
        hypothesis = self.model(X)
        y_predicted = torch.where(hypothesis < 0.5, 0, 1).numpy()     
        # ==========================================================

        return y_predicted

