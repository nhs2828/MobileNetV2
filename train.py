import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from archi import *
from torch.utils.tensorboard import SummaryWriter


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# set up data CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# set up model, loss and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MyNet(nb_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
EPOCH = 50

# train loop
layout = {
    "Toto": {
        "loss": ["Multiline", ["loss/train", "loss/test"]],
        "acc": ["Multiline", ["acc/train", "acc/test"]],
    },
}
writer = SummaryWriter()
writer.add_custom_scalars(layout)

for epoch in range(EPOCH):  # loop over the dataset multiple times

    train_loss = 0.0
    train_acc = 0.0
    nb_img = 0
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()
        nb_img += len(labels)
    writer.add_scalar("loss/train", train_loss/len(trainloader), epoch)
    writer.add_scalar("acc/train", train_acc/nb_img, epoch)
    # val
    net.eval()
    test_loss = 0.0
    test_acc = 0.0
    nb_img = 0
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # print statistics
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_acc += (predicted == labels).sum().item()
        nb_img += len(labels)
    writer.add_scalar("loss/test", test_loss/len(testloader), epoch)
    writer.add_scalar("acc/test", test_acc/nb_img, epoch)

writer.close()
print('Finished Training')