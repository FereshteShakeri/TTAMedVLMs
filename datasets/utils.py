import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import copy
import time
from torchvision.transforms import Resize
from torchvision import transforms
import collections.abc
from torch.utils.data import Dataset as _TorchDataset
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import Subset


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


class CopyDict():
    def __call__(self, data):
        d = copy.deepcopy(data)
        return d
    

def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = '' # the directory where the dataset is stored
    domains = [] # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x # labeled training data
        self._train_u = train_u # unlabeled training data (optional)
        self._val = val # validation data (optional)
        self._test = test # test data

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False,
                 return_img0=False, k_tfm=1):
        self.data_source = data_source
        self.transform = transform # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


def build_data_loader(
    data_source=None,
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None,
    pin_memory=True,
    drop_last=True
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(data_loader) > 0

    return data_loader

# Function lo load an image from data dict, and scale to target size
def load_image(image_path, size, canvas):

    # Read image
    if "dcm" in image_path:
        import pydicom
        dicom = pydicom.dcmread(image_path)
        img = np.array(dicom.pixel_array, dtype=float)
    else:
        img = Image.open(image_path)
        max_size = max(img.size)
        scale = max_size / size[0]
        img.draft('L', (img.size[0] / scale, img.size[1] // scale))
        img = np.asarray(img, dtype=float)

    # Scale intensity
    img /= 255.

    # Add channel
    img = np.expand_dims(img, 0)

    # Resize image
    img = torch.tensor(img)
    if not canvas or (img.shape[-1] == img.shape[-2]):
        img = Resize(size)(img)
    else:
        sizes = img.shape[-2:]
        max_size = max(sizes)
        scale = max_size / size[0]
        img = Resize((int(img.shape[-2] / scale), int((img.shape[-1] / scale)))).cuda()(img.cuda())
        img = torch.nn.functional.pad(img,
                                      (0, size[0] - img.shape[-1], 0, size[1] - img.shape[-2], 0, 0))
    img = img.cpu().numpy()
    return img


class LoadImage():

    def __init__(self, size=(224, 224), canvas=True, total_samples=300000, memory_cache=0.0):
        self.size = size
        self.canvas = canvas
        self.counter = 0
        self.total_samples = total_samples
        self.memory_cache = memory_cache
        
        # Define transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure it matches self.size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])
        ])

    def __call__(self, data):

        # If we are using cache memory
        if self.memory_cache > 0.0:
            # Check if image has already been loaded and pre-processed
            if "cache" in data.keys():
                d = copy.deepcopy(data)
                d["image"] = np.float32(d["image"]) / 255.
            # Otherwise, load image
            else:
                img = load_image(data['image_path'], self.size, self.canvas)
                # Check for space in cache memory
                if self.counter < (self.total_samples*self.memory_cache):
                    self.counter += 1
                    data["image"], data["cache"] = np.uint8((img * 255)), True
                d = copy.deepcopy(data)
                d["image"] = img
        else:
            img = load_image(data['image_path'], self.size, self.canvas)
            d = copy.deepcopy(data)
            d["image"] = img

        """
        # Add channels to grayscale image
        if d["image"].shape[0] == 1:
            d["image"] = np.repeat(d["image"], 3, 0)
        # Convert image to a PyTorch tensor with dtype=torch.float32
        d["image"] = torch.tensor(d["image"], dtype=torch.float32)
        """
        # Fix the shape issue
        img_array = d["image"]
        
        if isinstance(img_array, np.ndarray):
            # Ensure the array shape is [H, W] or [H, W, C]
            if img_array.ndim == 3 and img_array.shape[0] == 1:  # Shape like (1, H, W)
                img_array = img_array.squeeze(0)  # Remove the first dimension -> (H, W)
            elif img_array.ndim == 2:  # If still grayscale, add a channel dimension
                img_array = np.stack([img_array] * 3, axis=-1)  # Convert (H, W) -> (H, W, 3)
            elif img_array.ndim == 3 and img_array.shape[-1] == 1:  # Shape like (H, W, 1)
                img_array = np.repeat(img_array, 3, axis=-1)  # Convert to (H, W, 3)
        
        # Convert NumPy array to PIL Image
        d["image"] = Image.fromarray((img_array * 255).astype(np.uint8))  # Ensure uint8 format

        # Apply transformation
        d["image"] = self.transform(d["image"])

        return d
    
    
class LoadImageRetina():
    def __init__(self, target="image_path"):
        self.target = target
        """
        Load, organize channels, and standardize intensity of images.
        """

    def __call__(self, data):
        # Read image
        img = np.array(Image.open(data[self.target]), dtype=float)
        if np.max(img) > 1:
            img /= 255

        # channel first
        if len(img.shape) > 2:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, 0)

        if img.shape[0] > 3:
            img = img[1:, :, :]
        if "image" in self.target:
            if img.shape[0] < 3:
                img = np.repeat(img, 3, axis=0)

        data[self.target.replace("_path", "")] = img
        return data


class ImageScaling():
    
    """
    Method for image scaling. It includes two options: scaling from canvas, to avoid image distortions,
    and regular scaling trough resizing.
    """

    def __init__(self, size=(224, 224), canvas=True, target="image"):
        self.size = size
        self.canvas = canvas
        self.target = target

        self.transforms = torch.nn.Sequential(
            Resize(self.size),
        )

    def __call__(self, data):
        img = torch.tensor(data[self.target])
        if not self.canvas or (img.shape[-1] == img.shape[-2]):
            img = self.transforms(img)
        else:
            sizes = img.shape[-2:]
            max_size = max(sizes)
            scale = max_size/self.size[0]
            img = Resize((int(img.shape[-2]/scale), int((img.shape[-1]/scale))))(img)
            img = torch.nn.functional.pad(img, (0, self.size[0] - img.shape[-1], 0, self.size[1] - img.shape[-2], 0, 0))

        data[self.target] = img
        return data
    
        
class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)

def get_experiment_setting(experiment):

    # Transferability for classification
    if experiment == "chexpert_5x200":
        setting = {"experiment": "chexpert_5x200",
                   "targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "classes": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "datasets/configs/chexpert_5x200.csv",
                   "base_samples_path": "/data/CheXpert/CheXpert-v1.0/"
        }
    elif experiment == "mimic_5x200":
        setting = {"experiment": "mimic_5x200",
                   "targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "classes": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "datasets/configs/mimic_5x200.csv",
                   "base_samples_path": "/data/MIMIC-CXR-2/2.0.0/"
        }
    elif experiment == "rsna_pneumonia_test":
        setting = {"experiment": "RSNA_pneumonia_test",
                   "targets": ["Normal", "Pneumonia"],
                   "classes": ["Normal", "Pneumonia"],
                   "dataframe": "datasets/configs/rsna_pneumonia_test.csv",
                   "base_samples_path": "/data/RSNA_PNEUMONIA/"
                   }
    elif experiment == "02_MESSIDOR":
        setting = {"dataframe": "datasets/configs/" + "02_MESSIDOR.csv",
                   "task": "classification",
                   "targets": {"no diabetic retinopathy": 0, "mild diabetic retinopathy": 1,
                               "moderate diabetic retinopathy": 2, "severe diabetic retinopathy": 3,
                               "proliferative diabetic retinopathy": 4},
                   "classes": ["无糖尿病视网膜病变", "轻度糖尿病视网膜病变", "中度糖尿病视网膜病变",
                               "严重糖尿病视网膜病变", "增生性糖尿病视网膜病变"],
                   "base_samples_path": "/Medical/"
        }
    elif experiment == "13_FIVES":
        setting = {"dataframe": "datasets/configs/" + "13_FIVES.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "age related macular degeneration": 1, "diabetic retinopathy": 2,
                               "glaucoma": 3},
                    "classes": ["正常眼底", "年龄相关性黄斑变性", "糖尿病视网膜病变","青光眼"],
                   "base_samples_path": "/Medical/"}
    elif experiment == "08_ODIR200x3":
        setting = {"dataframe": "datasets/configs/" + "08_ODIR200x3.csv",
                   "task": "classification",
                   "classes": ["正常眼底", "病理性近视", "白内障"],
                   "targets": {"normal": 0, "pathologic myopia": 1, "cataract": 2},
                   "base_samples_path": "/Medical/"}
    else:
        setting = None
        print("Experiment not prepared...")

    return setting

