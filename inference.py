import torch
from image_loader import *
from ANET import *

net = ANet()
net.load_state_dict(torch.load('params_299.pkl', map_location={'cuda:2':'cuda:0'}))

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
imagelog = torch.log(image)

deshadow = torch.add(imagelog,-1, result)
deshadow = torch.exp(deshadow)
des = to_pil_image(deshadow[0])
des.show()
