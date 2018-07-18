import torch
from image_loader import *
from SNet import *
import torch.utils.data as data
import os

#hyper parameters
BATCH_SIZE = 1
BASE_LR = 1e-3
G_LR = 1e-5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0,2,3,4]

normalization_mean = torch.tensor([0.485, 0.456, 0.406])
normalization_std = torch.Tensor([0.229, 0.224, 0.225])


def make_label(image, label):
    imagelog = torch.log(image + 1).to(device)
    labellog = torch.log(label + 1).to(device)
    target = torch.abs(torch.add(imagelog, -1, labellog)).to(device)
    target *= 3
    return target


def Normalization(input, mean, std):
    m = torch.tensor(mean).view(-1, 1, 1).to(device)
    s = torch.tensor(std).view(-1, 1, 1).to(device)
    output = (input - m) / s
    return output



def single_gpu_train():
    dataset = mytraindata("./data/image", "./data/label", True, True)
    data_loader = data.DataLoader(dataset, batch_size= BATCH_SIZE)

    net = SNet()
    net = net.to(device)
    print(net)

    gparam = list(map(id, net.features.parameters()))
    base_param = filter(lambda p: id(p) not in gparam, net.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_param},
        {'params': net.features.parameters(), 'lr': G_LR}], lr = BASE_LR, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(100000):
        for i, data in enumerate(data_loader, 0):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            target = make_label(image, label)

            norm_image = Normalization(image, normalization_mean, normalization_std)
            prediction = net(norm_image)
            loss = criterion(prediction, target)

            print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 99:
            model_name = os.path.join('model/model_%d.pkl' % epoch)
            torch.save(net.state_dict(), model_name)



def multi_gpu_train():
    dataset = mytraindata("./data/image", "./data/label", True, True)
    data_loader = DATA.DataLoader(dataset, batch_size=BATCH_SIZE)
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).cuda(device_ids[0])
    normalization_std = torch.Tensor([0.229, 0.224, 0.225]).cuda(device_ids[0])

    net = SNet()

    net.cuda(device_ids[0])
    net = nn.DataParallel(net, device_ids=device_ids)

    gparam = list(map(id, net.module.features.parameters()))
    base_param = filter(lambda p: id(p) not in gparam, net.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_param},
        {'params': net.module.features.parameters(), 'lr': G_LR}], lr=BASE_LR, momentum=0.9, weight_decay=5e-4)

    criterion = torch.nn.MSELoss()
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    for epoch in range(100000):
        for i, data in enumerate(data_loader, 0):
            image, label = data
            target = make_label(image, label)
            norm_image = Normalization(image, normalization_mean, normalization_std)

            img = Variable(norm_image, requires_grad=True).cuda(device_ids[0])
            tar = Variable(target).cuda(device_ids[0])

            prediction = net(img)
            loss = criterion(prediction, tar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.module.step()

            print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))

        if epoch % 1 == 0:
            model_name = os.path.join("./model/model_%d.pkl" % epoch)
            torch.save(net, model_name)


single_gpu_train()
