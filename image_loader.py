import torch.utils.data as data
import os
import os.path
import glob
import torchvision.transforms as transforms
from PIL import Image


def make_dataset(img_path, label_path):
    dataset = []
    for img in glob.glob(os.path.join(img_path, '*.jpg')):
        basename = os.path.basename(img)
        image = os.path.join(img_path, basename)
        label = os.path.join(label_path, basename)
        dataset.append([image, label])
    return dataset


class mytraindata(data.Dataset):
    def __init__(self, img_path, label_path, transform=None, rescale=None):
        super(mytraindata, self).__init__()
        self.train_set_path = make_dataset(img_path, label_path)
        self.transform = transform
        self.rescale = rescale

    def __getitem__(self, item):
        img_path, label_path = self.train_set_path[item]
        image = Image.open(img_path)
        label = Image.open(label_path)
        transform = transforms.ToTensor()
        if self.rescale:
            image = image.resize((224, 224))
            label = label.resize((224, 224))
        if self.transform:
            image = transform(image)
            label = transform(label)
        return image, label


    def __len__(self):
        return len(self.train_set_path)
