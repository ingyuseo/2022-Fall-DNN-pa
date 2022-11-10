import numpy as np

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch
        # ========================= EDIT HERE ==========================

        # Train should be done for 'epochs' times.
        # The function 'train' should return the loss of final epoch.
        # Weights are updated through the optimizer, not directly within 'train' function.        
        # Perceptron updates only with samples whose predictions are wrong.
        train_size = x.shape[0]
    
        for epoch in range(epochs):
            loss =0

            #1. Shuffle data (x, y)
            np.random.seed(12345)
            permutation = list(np.random.permutation(train_size))

            x_shuffled = x[permutation, :]
            y_shuffled = y[permutation]

            #2. Train using mini-batch
            minibatches = train_size//batch_size 
            for iter in range(minibatches):
                batch_x = x_shuffled[iter*batch_size: (iter+1)*batch_size]
                batch_y = y_shuffled[iter*batch_size : (iter+1)*batch_size]

                batch_loss = 0
                grad = np.zeros(shape=(self.num_features, 1))
                prediction = (batch_x@self.W).flatten()

                cost = prediction*batch_y

                for idx in range(batch_size):
                    if cost[idx] < 0:
                        batch_loss -= cost[idx]
                        grad = grad - batch_x[idx].reshape(self.num_features,1)*batch_y[idx]

                loss += batch_loss
                #when loss in mini-batch is 0, take next mini-batch.
                if batch_loss == 0:
                    continue
                
                self.W = optim.update(self.W, grad, lr)

        # ============================================================
        return loss

    def forward(self, x):
        y_predicted = None
        # ========================= EDIT HERE ========================
        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the output is positive or equal to 0.
        # Otherwise, it predicts as 0

        result = x@self.W
        y_predicted = np.where(result >=0, 1, -1)

        # ==========================================================

        return y_predicted

