
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 2  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
#LR = 0.05  # learning rate
LR = 0.01  # learning rate
DOWNLOAD_MNIST = False

# Mnist digits dataset
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example
# print(train_data.train_data.size())  # (60000, 28, 28)
# print(train_data.train_labels.size())  # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
train_x = torch.unsqueeze(train_data.train_data, dim=1).type(torch.FloatTensor)
# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
         :5000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:5000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(  # 卷积层
                in_channels=1,  # input height 灰度图只有一层
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size 5个像素点 5*5
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation 激励层
            nn.MaxPool2d(kernel_size=2),  # 池化层 choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # fully connected layer, output 10 classes
        self.fc2 = nn.Linear(128, 10)  # fully connected layer, output 10 classes
        self.fc3 = nn.Linear(10, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output, x  # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
#optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)  # 打开SGD的开关;
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm

# training and testing

epoch_set = []
epoch_validation = []
epoch_train = []
epoch_accuracy = []

step_cnt = []
step_validation = []
step_train = []
step_accuracy = []

cnt = 0
Accuracy_result=0.0

for epoch in range(EPOCH):
    last_loss = 0.0
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        cnt += 1
        output = cnn(b_x)[0]  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        last_loss = loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            validation_loss = loss_func(test_output, test_y)
            step_cnt.append(cnt)
            step_train.append(loss.data.numpy())
            step_validation.append(validation_loss.data.numpy())
            step_accuracy.append(accuracy)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(),
                  '| validation_loss: %.4f' % validation_loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            # if(loss.data.numpy()<0.0015):
            #     break

    test_output1, last_layer1 = cnn(test_x)
    pred_y1 = torch.max(test_output1, 1)[1].data.numpy()
    accuracy1 = float((pred_y1 == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    validation_loss1 = loss_func(test_output1, test_y)
    epoch_set.append(epoch + 1)
    epoch_train.append(last_loss.data.numpy())
    epoch_validation.append(validation_loss1.data.numpy())
    epoch_accuracy.append(accuracy1)
    print('Epoch: ', epoch, '| train loss: %.4f' % last_loss.data.numpy(),
          '| validation_loss: %.4f' % validation_loss1.data.numpy(), '| test accuracy: %.2f' % accuracy1)
    Accuracy_result=accuracy1


plt.title('Train loss and validation loss')
plt.plot(step_cnt, step_train, color='green', label='Train loss')
plt.plot(step_cnt, step_validation, color='blue', label='Validation loss')
plt.xlabel('Training count')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Train_and_validation_5_7_adam.png', dpi=900)
plt.show()

plt.title('Epoch and validation loss')
plt.plot(epoch_set, epoch_validation, color='blue', label='Validation loss')
plt.xlabel('Epoch count')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Epoch_and_validation_5_7_adam.png', dpi=900)
plt.show()

plt.title('Epoch and accuracy')
plt.plot(epoch_set, epoch_accuracy, color='blue', label='Accuracy')
plt.xlabel('Epoch count')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Epoch_and_accuracy_5_7_adam.png', dpi=900)
plt.show()

plt.title('Training count and accuracy')
plt.plot(step_cnt, step_accuracy, color='blue', label='Accuracy')
plt.xlabel('Training count')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Training count_and_accuracy_5_7_adam.png', dpi=900)
plt.show()

print('LR:',LR)
print('EPOCH',EPOCH)
print('Adam')
print('Accuracy',Accuracy_result)
