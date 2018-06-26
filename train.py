from torch.nn import MSELoss
from image_loader import *
from AGNET import *
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models


# define a function to get the target
def makeLabel(data, label):
    datalog = torch.log(data+1)
    labellog = torch.log(label+1)
    target = torch.abs(torch.add(datalog, -1, labellog))
    return target

# define a function to initial the weights with Gaussian params of mean=0, std = 0.001
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0, std=0.001)
        torch.nn.init.constant_(m.bias, 0.1)

# load the data, you should change the data relative address in "image_load.py"
batch_size = 1
dataset = mytraindata(".", transform=True, train=True, rescale=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# create a GNet
net = GNet()
# initial the params 
net.apply(weights_init)
# using the vgg16 pre-trained model
vgg16 = models.vgg16(pretrained=True)
#print(vgg16)
pretrained_dict = vgg16.state_dict()
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

print(net)
# if cuda is available, put the net on the gpu
if torch.cuda.is_available():
    net = net.cuda()
   
# define the loss function
criterion = MSELoss()

# setting a optimizer with different learning rate
gparam = list(map(id, net.features.parameters()))
base_params = filter(lambda p : id(p) not in gparam, net.parameters())
optimizer = optim.SGD([
            {'params': base_params},
            {'params': net.features.parameters(), 'lr': 0.00001}], lr=0.0001, momentum=0.9)

print(net.state_dict())

# train
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
        loss = criterion(outputs, target)
      
        print(loss, i, epoch)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        if i % 500 == 499:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0
    if epoch % 100 == 99:
        model_name = os.path.join('model/params_%d.pkl' % epoch)
        torch.save(net.state_dict(), model_name)
    #torch.save(net.state_dict(), 'params.pkl')
#    print("finished training")

