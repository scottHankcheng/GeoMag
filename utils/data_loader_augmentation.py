from torchvision.datasets.folder import *
import numpy as np
import torch
from torchvision import transforms
import random

class PairedRandomTransform:
    def __init__(self, transform_ops):
        self.transform_ops = transform_ops
    
    def __call__(self, imgs):
        seed = random.randint(0, 2**32)
        transformed_imgs = []
        
        for img in imgs:
            random.seed(seed)
            torch.manual_seed(seed)
            pil_img = transforms.ToPILImage()(img)
            
            for op in self.transform_ops:
                pil_img = op(pil_img)
            
            transformed_imgs.append(np.array(pil_img))
        
        return transformed_imgs


class ImageFromFolder(ImageFolder):
    def __init__(self, root, num_data=100000, preprocessing=False, transform=None, target_transform=None,
                 loader=default_loader, augmentation=False):

        mag = np.loadtxt(os.path.join(root, 'train_mf.txt'))

        imgs = [(os.path.join(root,'amplified','%06d.png'%(i)),
                 os.path.join(root,'frameA','%06d.png'%(i)),
                 os.path.join(root,'frameB','%06d.png'%(i)),
                 os.path.join(root,'frameC','%06d.png'%(i)),
                 mag[i]) for i in range(num_data)]

        self.root = root
        self.imgs = imgs
        self.samples = self.imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.preproc = preprocessing
        self.augmentation = augmentation
        
        if self.augmentation:
            self.aug_transform = PairedRandomTransform([
                transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=10)
            ])


  
    def __getitem__(self, index):
        pathAmp, pathA, pathB, pathC, target = self.samples[index]
        sampleAmp, sampleA, sampleB, sampleC = np.array(self.loader(pathAmp)), np.array(self.loader(pathA)), np.array(self.loader(pathB)), np.array(self.loader(pathC))
      
        if self.augmentation:
            sampleAmp, sampleA, sampleB, sampleC = self.aug_transform([sampleAmp, sampleA, sampleB, sampleC])
        
        sampleAmp = sampleAmp/127.5 - 1.0
        sampleA = sampleA/127.5 - 1.0
        sampleB = sampleB/127.5 - 1.0
        sampleC = sampleC/127.5 - 1.0

        if self.preproc:
            sampleAmp = preproc_poisson_noise(sampleAmp)
            sampleA = preproc_poisson_noise(sampleA)
            sampleB = preproc_poisson_noise(sampleB)
            sampleC = preproc_poisson_noise(sampleC)

        sampleAmp, sampleA, sampleB, sampleC = torch.from_numpy(sampleAmp), torch.from_numpy(sampleA), torch.from_numpy(sampleB), torch.from_numpy(sampleC)
        sampleAmp = sampleAmp.float()
        sampleA = sampleA.float()
        sampleB = sampleB.float()
        sampleC = sampleC.float()

        target = torch.from_numpy(np.array(target)).float()

        sampleAmp = sampleAmp.permute(2,0,1)
        sampleA = sampleA.permute(2,0,1)
        sampleB = sampleB.permute(2,0,1)
        sampleC = sampleC.permute(2,0,1)

        return sampleAmp, sampleA, sampleB, sampleC, target


def preproc_poisson_noise(image):
    nn = np.random.uniform(0, 0.3)
    n = np.random.normal(0.0, 1.0, image.shape)
    n_str = np.sqrt(image + 1.0) / np.sqrt(127.5)
    return image + nn * n * n_str

class ImageFromFolderTest(ImageFolder):
    def __init__(self, root, mag=10.0, mode='static', num_data=300, preprocessing=False, transform=None, target_transform=None, loader=default_loader):
        if mode=='static':
            imgs = [(root+'_%06d.png'%(1),
                     root+'_%06d.png'%(i+2),
                     mag) for i in range(num_data)]
        elif mode=='dynamic':
            imgs = [(root+'_%06d.png'%(i+1),
                     root+'_%06d.png'%(i+2),
                     mag) for i in range(num_data)]
        else:
            raise ValueError("Unsupported modes %s"%(mode))

        self.root = root
        self.imgs = imgs
        self.samples = self.imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.preproc = preprocessing

    def __getitem__(self, index):
        pathA, pathB, target = self.samples[index]
        sampleA, sampleB = np.array(self.loader(pathA)), np.array(self.loader(pathB))
      
        sampleA = sampleA/127.5 - 1.0
        sampleB = sampleB/127.5 - 1.0

        if self.preproc:
            sampleA = preproc_poisson_noise(sampleA)
            sampleB = preproc_poisson_noise(sampleB)

        sampleA, sampleB = torch.from_numpy(sampleA), torch.from_numpy(sampleB)
        sampleA = sampleA.float()
        sampleB = sampleB.float()

        target = torch.from_numpy(np.array(target)).float()

        sampleA = sampleA.permute(2,0,1)
        sampleB = sampleB.permute(2,0,1)

        return sampleA, sampleB, target