from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch
import torch.utils.data as data
import warnings
from .utils import download_url, check_integrity
import random
import torchvision.datasets as datasets
import yaml
from decord import VideoReader, cpu
import pandas as pd
# from ucf101 import spatial_sampling
import dataloaders.video_transforms as video_transforms 
import dataloaders.volume_transforms as volume_transforms
from torchvision import transforms

class iUCF101(data.Dataset):
    
    def __init__(self, root,
                train=True, transform=None,
                download_flag=False, lab=True, swap_dset = None, 
                tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5,args=None,anno_path=None):

        # process rest of args
        self.root = root 
        self.anno_path = anno_path
        self.transform = transform
        self.train = train  # training set or test set
        self.crop_size = 224
        self.short_side_size = 224
        self.keep_aspect_ratio = True
        self.num_segment = 8
        self.new_height = 256
        self.new_width = 320
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag
        self.args = args
        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))
        if not self.train:
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        # self.data = np.asarray(self.data)
        # self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)
            # sample
            n_data = len(self.targets)
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
            else:
                self.data = self.data[int(0.8*n_data):]
                self.targets = self.targets[int(0.8*n_data):]

            # train set
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

            # val set
            else:
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))


        # else
        else:
            self.archive = []
            domain_i = 0
            for task in self.tasks:
                if True:
                    locs = np.isin(self.targets, task).nonzero()[0]                       
                    task_data = [self.data[i] for i in locs]
                    task_target = [self.targets[i] for i in locs]
                    self.archive.append((task_data.copy(), task_target.copy()))

        # if self.train:
        #     self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        # img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        if self.train:
            scale_t = 1
            sample = self.data[index]
            target = self.targets[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.data[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
            buffer = self._aug_frame(buffer, self.args)
            return buffer, self.class_mapping[target],self.t
        else:
            sample = self.data[index]
            target = self.targets[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.data[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.class_mapping[target],self.t
        #!original    
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        # return img, self.class_mapping[target], self.t

    def load_dataset(self, t, train=True):
        
        if train:
            self.data, self.targets = self.archive[t] 
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

    def append_coreset(self, only=False, interp=False):
        len_core = len(self.coreset[0])
        if self.train and (len_core > 0):
            if only:
                self.data, self.targets = self.coreset
            else:
                len_data = len(self.data)
                sample_ind = np.random.choice(len_core, len_data)
                self.data = np.concatenate([self.data, self.coreset[0][sample_ind]], axis=0)
                self.targets = np.concatenate([self.targets, self.coreset[1][sample_ind]], axis=0)

    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        
        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def load(self):
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.data = list(cleaned.values[:, 0])
        self.targets = list(cleaned.values[:, 1])

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',#args.aa,
            interpolation='bicubic',
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.dataset == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        return buffer
    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = os.path.join(self.root,sample)+'.avi'
        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        # if self.mode == 'test':
        #     all_index = []
        #     tick = len(vr) / float(self.num_segment)
        #     all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
        #                        [int(tick * x) for x in range(self.num_segment)]))
        #     while len(all_index) < (self.num_segment * self.test_num_segment):
        #         all_index.append(all_index[-1])
        #     all_index = list(np.sort(np.array(all_index))) 
        #     vr.seek(0)
        #     buffer = vr.get_batch(all_index).asnumpy()
        #     return buffer

        # handle temporal segments
        average_duration = len(vr) // self.num_segment
        all_index = []
        if average_duration > 0:
            all_index += list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                        size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index += list(np.zeros((self.num_segment,)))
        all_index = list(np.array(all_index)) 
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

class iDataset(data.Dataset):
    
    def __init__(self, root,
                train=True, transform=None,
                download_flag=False, lab=True, swap_dset = None, 
                tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:
            
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

            # sample
            n_data = len(self.targets)
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
            else:
                self.data = self.data[int(0.8*n_data):]
                self.targets = self.targets[int(0.8*n_data):]

            # train set
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

            # val set
            else:
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        # else
        else:
            self.archive = []
            domain_i = 0
            for task in self.tasks:
                if True:
                    locs = np.isin(self.targets, task).nonzero()[0]
                    self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        if self.train:
            self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t

    def load_dataset(self, t, train=True):
        
        if train:
            self.data, self.targets = self.archive[t] 
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

    def append_coreset(self, only=False, interp=False):
        len_core = len(self.coreset[0])
        if self.train and (len_core > 0):
            if only:
                self.data, self.targets = self.coreset
            else:
                len_data = len(self.data)
                sample_ind = np.random.choice(len_core, len_data)
                self.data = np.concatenate([self.data, self.coreset[0][sample_ind]], axis=0)
                self.targets = np.concatenate([self.targets, self.coreset[1][sample_ind]], axis=0)

    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        
        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def load(self):
        pass

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class iCIFAR10(iDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iDataset Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    im_size=32
    nch=3

    def load(self):

        # download dataset
        if self.download_flag:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or self.validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iCIFAR10 Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    im_size=32
    nch=3

class iIMAGENET_R(iDataset):
    
    base_folder = 'imagenet-r'
    im_size=224
    nch=3
    def load(self):

        # load splits from config file
        if self.train or self.validation:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_train.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_test.yaml', 'r'), Loader=yaml.Loader)
        self.data = data_config['data']
        self.targets = data_config['targets']

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = jpg_image_to_array(img_path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

class iDOMAIN_NET(iIMAGENET_R):
    base_folder = 'DomainNet'
    im_size=224
    nch=3
    def load(self):
        
        # load splits from config file
        if self.train or self.validation:
            data_config = yaml.load(open('dataloaders/splits/domainnet_train.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('dataloaders/splits/domainnet_test.yaml', 'r'), Loader=yaml.Loader)
        self.data = data_config['data']
        self.targets = data_config['targets']

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:      
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr









def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames