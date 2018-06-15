from torch.nn import MSELoss
from image_loader import *
from AGNET import *
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models

def makeLabel(data, label):
    datalog = torch.log(torch.clamp(data, min=1e-8))
    labellog = torch.log(torch.clamp(label,min=1e-8))
    target = datalog - labellog
    return target


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0, std=0.001)
        torch.nn.init.constant_(m.bias, 0.1)


batch_size = 1
dataset = mytraindata(".", transform=True, train=True, rescale=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

net = GNet()
net.apply(weights_init)
vgg16 = models.vgg16(pretrained=True)
#print(vgg16)
pretrained_dict = vgg16.state_dict()
model_dict = net.state_dict()
rest_dict = net.state_dict()
rest_dict = {k:v for k, v in rest_dict.items() if k not in pretrained_dict}
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
print(net)
if torch.cuda.is_available():
    net = net.cuda()
    
criterion = MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
print(net.state_dict())


for epoch in range(100000):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        target = makeLabel(inputs, labels)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        predict = torch.log(torch.clamp(outputs, min=1e-8))       
        loss = criterion(predict, target)
      
        print(loss, i, epoch)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    if epoch % 100 == 99:
        model_name = os.path.join('model/params_%d.pkl' % epoch)
        torch.save(net.state_dict(), model_name)
    #torch.save(net.state_dict(), 'params.pkl')
#    print("finished training")

