from torch.utils.data import Subset
from PIL import Image
# from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np

import torchvision.transforms as transforms
import os
import torchvision
from torchvision.datasets import ImageFolder

class Casting_Dataset(TorchvisionDataset):
    def __init__(self, root:str,normal_class = 0):
        super().__init__(root)
        assert normal_class in [0,1]
        self.root = root
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = normal_class
        self.outlier_classes = 1-normal_class
        
        
        train_transforms = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=(0.9, 1.0)),
            transforms.ToTensor()
            ,transforms.Normalize((.5642, 0.5642, 0.5642), (0.2369, 0.2369, 0.2369))
        #     ,transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(size=(32, 32)),
        #     transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ,transforms.Normalize((.5642, 0.5642, 0.5642), (0.2369, 0.2369, 0.2369))
        #     ,transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        raw_transforms = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=(0.9, 1.0)),
            transforms.ToTensor()
        #     ,transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
	
        
        
        
        # Pre-computed min and max values (after applying GCN) from train data per class
#         min_max = [(-0.8826567065619495, 9.001545489292527),
#                    (-0.6661464580883915, 20.108062262467364),
#                    (-0.7820454743183202, 11.665100841080346),
#                    (-0.7645772083211267, 12.895051191467457),
#                    (-0.7253923114302238, 12.683235701611533),
#                    (-0.7698501867861425, 13.103278415430502),
#                    (-0.778418217980696, 10.457837397569108),
#                    (-0.7129780970522351, 12.057777597673047),
#                    (-0.8280402650205075, 10.581538445782988),
#                    (-0.7369959242164307, 10.697039838804978)]

        # can apply min-max scaling intead of normalize
        train_dataset = MyImage(root=os.path.join(self.root,'train'), transform=train_transforms)
        test_dataset = MyImage(root=os.path.join(self.root,'test'), transform=test_transforms,train=False,raw_transform = raw_transforms)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_dataset.targets, self.normal_classes)
        self.train_set = Subset(train_dataset, train_idx_normal)

        self.test_set = test_dataset

class MyImage(ImageFolder):
    def __init__(self,train=True,raw_transform=None,*args,**kwargs):
        super(MyImage,self).__init__(*args,**kwargs)
        self.train = train
        self.raw_transform = raw_transform
        self.make_dataarray()
		
    def make_dataarray(self):
        arrl = []
        for fpth,_ in self.imgs:
            #print(self.loader(fpth))
            if self.raw_transform is not None:
                arrl.append(self.raw_transform(self.loader(fpth)).detach().cpu().numpy()[np.newaxis])
            else:
                if self.transform is not None:
                    arrl.append(self.transform(self.loader(fpth)).detach().cpu().numpy()[np.newaxis])
        if self.train:
            self.train_data = np.concatenate(arrl)
        else:
            self.test_data = np.concatenate(arrl)
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target,index

        
