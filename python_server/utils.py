from PIL import Image
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps
import os
import shutil
import pandas as pd
#from torch._utils import _accumulate
from itertools import accumulate as _accumulate
from torch import randperm
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.legend

DATA_ROOT = os.getenv("DATA_ROOT", "python_server/data")
ALLOW_DATASET_DOWNLOAD = os.getenv("ALLOW_DATASET_DOWNLOAD", "false").lower() in ("1", "true", "yes")

def load_dataset(args, dataset, cluster=None, mode = 'target', max_num = 2000):
    kwargs = {'num_workers': 2, 'pin_memory': True}
    # load trainset and testset
    
    if mode == 'shadow' or mode == 'ChangeDataSize':
        if dataset == 'GTSRB':
            transform = transforms.Compose([Rand_Augment(), transforms.Resize((64,64)), transforms.ToTensor()])
        else:
            transform = transforms.Compose([Rand_Augment(), transforms.ToTensor()])
    else:
        if dataset == 'GTSRB':
            transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            
    if dataset == 'CIFAR10':
        whole_set = datasets.CIFAR10(DATA_ROOT, train=True, download=ALLOW_DATASET_DOWNLOAD, transform=transform)
        max_cluster = 3000
        test_size = 1000
    elif dataset == 'CIFAR100':
        whole_set = datasets.CIFAR100(DATA_ROOT, train=True, download=ALLOW_DATASET_DOWNLOAD, transform=transform)
        max_cluster = 7000
        test_size = 1000
    elif dataset == 'GTSRB':
        whole_set = datasets.ImageFolder(os.path.join(DATA_ROOT, 'GTSRB'), transform= transform)
        max_cluster = 600
        test_size = 500
    elif dataset == 'Face':
        whole_set = datasets.ImageFolder(os.path.join(DATA_ROOT, 'lfw'), transform=transform)
        max_cluster = 350
        test_size = 100
    # elif dataset == 'TinyImageNet':
    #     whole_set = datasets.ImageFolder('data/tiny-imagenet-200/train', transform=transform)
    #     max_cluster = 30000
    #     test_size = 2000
    length = len(whole_set)
    if mode == 'target':
        train_size = cluster
        remain_size = length - train_size - test_size
        train_set, _, test_set = dataset_split(whole_set, [train_size, remain_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'shadow': 
        train_size = length - max_cluster - test_size
        _, train_set, test_set = dataset_split(whole_set, [max_cluster, train_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader#, test_loader
    elif mode == 'salem_unknown': 
        train_size = length - max_cluster - test_size
        salme_train = int(train_size * 0.5)
        salme_test = train_size - salme_train
        _, train_set, test_set, _ = dataset_split(whole_set, [max_cluster, salme_train, salme_test, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'salem_known': 
        salme_train = cluster
        salme_test = cluster
        rest_size =  length - max_cluster - test_size - cluster - cluster
        _, train_set, test_set, _, _ = dataset_split(whole_set, [max_cluster, salme_train, salme_test, rest_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'ChangeDataSize':
        train_size = cluster
        remain_size = length - max_cluster - train_size - test_size
        _, train_set, _, _ = dataset_split(whole_set, [max_cluster, train_size, remain_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader

    ###
    elif mode in ['adversary', 'radius']:
        mem_size = min([cluster, test_size, max_num])
        non_size = mem_size
        remain_size = length - mem_size - non_size
        mem_set, _, non_set = dataset_split(whole_set, [mem_size, remain_size, non_size])

        data_set = ConcatDataset([mem_set, non_set])
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)
        if mode == 'radius':
            return mem_set, non_set, transform
        else:
            return data_loader
     

def load_dataset_DataAug_AdvReg(args, dataset, cluster=None, defense=None):
    kwargs = {'num_workers': 2, 'pin_memory': True}
    # load trainset and testset
    if defense == 'DataAug':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]) # transforms.RandomErasing();  Rand_Augment()
    elif defense == 'AdvReg':
        transform = transforms.Compose([transforms.ToTensor()])

    if dataset == 'CIFAR10':
        whole_set = datasets.CIFAR10(DATA_ROOT, train=True, download=ALLOW_DATASET_DOWNLOAD, transform=transform)
        test_size = 1000

    length = len(whole_set)
    
    train_size = cluster
    remain_size = length - train_size - test_size


    train_set, _, _ = dataset_split(whole_set, [train_size, remain_size, test_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader

        


def dataset_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = list(range(sum(lengths)))
    np.random.seed(1)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class Rand_Augment():
    def __init__(self, Numbers=None, max_Magnitude=None):
        self.transforms = ['autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize',
                           'contrast', 'brightness', 'sharpness', 'shearX', 'shearY', 'translateX', 'translateY']
        if Numbers is None:
            self.Numbers = len(self.transforms) // 2
        else:
            self.Numbers = Numbers
        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        fillcolor = 128
        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation ,
            # it is no  need to obey  the value  in  autoaugment.py
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 0.2, 10),
            "translateY": np.linspace(0, 0.2, 10),
            "rotate": np.linspace(0, 360, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 231, 10),
            "contrast": np.linspace(0.0, 0.5, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.3, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,           
            "invert": [0] * 10
        }
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fill=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fill=fillcolor),
            "rotate": lambda img, magnitude: self.rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: img,
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def rand_augment(self):
        """Generate a set of distortions.
             Args:
             N: Number of augmentation transformations to apply sequentially. N  is len(transforms)/2  will be best
             M: Max_Magnitude for all the transformations. should be  <= self.max_Magnitude """

        M = np.random.randint(0, self.max_Magnitude, self.Numbers)

        sampled_ops = np.random.choice(self.transforms, self.Numbers)
        return [(op, Magnitude) for (op, Magnitude) in zip(sampled_ops, M)]

    def __call__(self, image):
        operations = self.rand_augment()
        for (op_name, M) in operations:
            operation = self.func[op_name]
            mag = self.ranges[op_name][M]
            image = operation(image, mag)
        return image

    def rotate_with_fill(self, img, magnitude):
        #  I  don't know why  rotate  must change to RGBA , it is  copy  from Autoaugment - pytorch
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

def fixed_seed(args):
    if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            cudnn.deterministic = True
##############################################
##############################################
##############################################
import torch.nn.init as init
import numpy as np

init_param = np.sqrt(2)
init_type = 'default'
def init_func(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            if init_type == 'normal':
                init.normal_(m.weight, 0.0, init_param)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=init_param)
            elif init_type == 'xavier_unif':
                init.xavier_uniform_(m.weight, gain=init_param)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
            elif init_type == 'kaiming_out':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_param)
            elif init_type == 'zero':
                init.zeros_(m.weight)
            elif init_type == 'one':
                init.ones_(m.weight)
            elif init_type == 'constant':
                init.constant_(m.weight, init_param)
            elif init_type == 'default':
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()

def save_code(path):
    os.makedirs(path + '/code', exist_ok=True)
    #files=os.listdir(args.logdir + '/code',) 
    #code_num = len([path for path in files if 'code' in path])
    #os.makedirs(args.logdir + '/code' + '/code_'+str(code_num), exist_ok=True)
    shutil.copyfile('main.py', path + '/code/main.py')
    shutil.copyfile('deeplearning.py', path + '/code/deeplearning.py')
    shutil.copyfile('classifier.py', path + '/code/classifier.py')
    shutil.copyfile('utils.py', path + '/code/utils.py')
    shutil.copyfile('plot.py', path + '/code/plot.py')
    shutil.copyfile('attack.py', path + '/code/attack.py')
