import torch
from image_loader import *
from AGNET import *

net = GNet()
#net.load_state_dict(torch.load('params_18899.pkl', map_location={'cuda:2':'cuda:0'}))
net.load_state_dict(torch.load('params_18899.pkl')

img_path = "./data/image/guangzhou_0901_00001.jpg"
image = Image.open(img_path)
image = image.resize((224, 224))
transform = transforms.ToTensor()
image = transform(image)
image = image.resize_(1,3,224,224)

result = net(image).cpu()

to_pil_image = transforms.ToPILImage()

res = to_pil_image(result[0])
res.show()
imagelog = torch.log(image+1)

deshadow = torch.add(imagelog, result)
deshadow = torch.exp(deshadow)-1
des = to_pil_image(deshadow[0])
des.show()
