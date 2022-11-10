import time
import numpy as np
from utils import load_mnist, load_emnist_part
from Answer import ClassifierModel, FCLayer, SoftmaxLayer, ReLU, LeakyReLU, ELU, EvaluationMetric
import matplotlib.pyplot as plt
import copy

np.random.seed(123)
np.tanh = lambda x: x

model = ClassifierModel()


"""
    Build model Architecture and do experiment.
    You can add layer as below examples.
    NOTE THAT, layers are executed in the order that you add.
    Enjoy.

    < Add Layers examples >
    - FC Layer
        model.add_layer('FC Example Layer', FCLayer(input_dim=1234, output_dim=123))
    - Softmax Layer
        model.add_layer('Softmax Layer', SoftmaxLayer())
"""

# mnist / emnist_part
dataset = 'emnist_part'

assert dataset in ['mnist', 'emnist_part']
if dataset == 'mnist':
    num_of_class = 10
else:
    num_of_class = 8

# Hyper-parameters
num_epochs = 20
learning_rate = 0.001
reg_lambda = 3*1e-4
print_every = 1

batch_size = 128

# Add layers
# model.add_layer('FC-1', FCLayer(784, 10))
# model.add_layer('ReLU', ReKU())
# model.add_layer('Softmax Layer', SoftmaxLayer()) 

# Add layers
model.add_layer('FC-1', FCLayer(784, 512))
model.add_layer('ReLU1', ELU())
model.add_layer('FC-2', FCLayer(512, 128))
model.add_layer('ReLU2', ELU())
model.add_layer('FC-4', FCLayer(128, num_of_class))
model.add_layer('Softmax Layer', SoftmaxLayer())

# =========================================================================

# Dataset
if dataset == 'mnist':
    x_train, y_train, x_test, y_test = load_mnist('./data')
else:
    x_train, y_train, x_test, y_test = load_emnist_part('./data')

x_train, x_test = np.squeeze(x_train), np.squeeze(x_test)

# Random 10% of train data as valid data
num_train = len(x_train)
perm = np.random.permutation(num_train)
num_valid = int(len(x_train) * 0.1)

valid_idx = perm[:num_valid]
train_idx = perm[num_valid:]

x_valid, y_valid = x_train[valid_idx], y_train[valid_idx]
x_train, y_train = x_train[train_idx], y_train[train_idx]

num_train, height, width = x_train.shape
num_class = y_train.shape[1]

model.summary()
time.sleep(1)

train_accuracy = []
valid_accuracy = []

best_epoch = -1
best_acc = -1
best_model = None

print('Training Starts...')
num_batch = int(np.ceil(num_train / batch_size))

for epoch in range(1, num_epochs + 1):
    # model Train
    start = time.time()
    epoch_loss = 0.0
    for b in range(num_batch):
        s = b * batch_size
        e = (b+1) * batch_size if (b+1) * batch_size < len(x_train) else len(x_train)

        x_batch = x_train[s:e]
        y_batch = y_train[s:e]

        loss = model.forward(x_batch, y_batch, reg_lambda)
        epoch_loss += loss

        model.backward(reg_lambda)
        model.update(learning_rate)

        print('[%4d / %4d]\t batch loss : %.4f' % (s, num_train, loss))

    end = time.time()
    lapse_time = end - start
    print('Epoch %d took %.2f seconds\n' % (epoch, lapse_time))

    if epoch % print_every == 0:
        # TRAIN ACCURACY
        prob = model.predict(x_train)
        pred = np.argmax(prob, -1).astype(int)
        true = np.argmax(y_train, -1).astype(int)

        correct = len(np.where(pred == true)[0])
        total = len(true)
        train_acc = correct / total
        train_accuracy.append(train_acc)

        # VAL ACCURACY
        prob = model.predict(x_valid)
        pred = np.argmax(prob, -1).astype(int)
        true = np.argmax(y_valid, -1).astype(int)

        correct = len(np.where(pred == true)[0])
        total = len(true)
        valid_acc = correct / total
        valid_accuracy.append(valid_acc)

        print('[EPOCH %d] Loss = %.5f' % (epoch, epoch_loss))
        print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)

        if best_acc < valid_acc:
            print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
            best_acc = valid_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)

print('Training Finished...!!')
print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))

# TEST ACCURACY
prob = best_model.predict(x_test)
pred = np.argmax(prob, -1).astype(int)
true = np.argmax(y_test, -1).astype(int)


correct = len(np.where(pred == true)[0])
total = len(true)
test_acc = correct / total

print('Test Accuracy at Best Epoch : %.2f' % (test_acc))

"""
    Draw a plot of train/valid accuracy.
    X-axis : Epoch
    Y-axis : train_accuracy & valid_accuracy
    Draw train_acc-epoch, valid_acc-epoch graph in 'one' plot.
"""
epochs = list(np.arange(1, num_epochs+1, print_every))

plt.plot(epochs, train_accuracy, label='Train Acc.')
plt.plot(epochs, valid_accuracy, label='Valid Acc.')

plt.title('Epoch - Train/Valid Acc.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

"""
    Calcuate precision, recall, and f-measure scores.
"""
print()
print('Precision, Recall, F-measure scores')
metric = EvaluationMetric(num_of_class)
scores = metric.multiclass_f_measure(pred, true, num_of_class)
print()
for key in scores.keys():
    print(f"{key}: {scores[key]:.4f}")

