import torch.utils.data as data
import os
import os.path
import glob
import torchvision.transforms as transforms
from PIL import Image


def make_dataset(root, train=True):
    dataset = []
    if train:
        dirgt = os.path.join(root, 'data/label')
        dirimg = os.path.join(root, 'data/image')

        for fGT in glob.glob(os.path.join(dirgt, '*.jpg')):
            # for k in range(45)
#            print(fGT)
            fName = os.path.basename(fGT)
            fImg = fName
 #           print(fName)
  #          print(fImg)
            dataset.append([os.path.join(dirimg, fImg), os.path.join(dirgt, fName)])
        print(dataset)
    return dataset



class mytraindata(data.Dataset):

    def __init__(self, root, transform=None, train=True, rescale=None):
        self.train = train
        self.transform = transform
        self.rescale = rescale
        if self.train:
            self.train_set_path = make_dataset(root, train)

    def __getitem__(self, idx):
        if self.train:
            img_path, label_path = self.train_set_path[idx]
            img = Image.open(img_path)
            if self.rescale:
                img = img.resize((224, 224))
            if self.transform:
                transform = transforms.ToTensor()
                img = transform(img)

            label = Image.open(label_path)
            if self.rescale:
                label = label.resize((224, 224))
            if self.transform:
                transform = transforms.ToTensor()
                label = transform(label)
            return img, label

    def __len__(self):
        return len(self.train_set_path)
