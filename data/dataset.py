# from https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py
import math
import random
import json
import logging
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)


def _numeric_name_key(file_name):
    stem = os.path.splitext(file_name)[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F.get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w
 
class CustomDataset(Dataset):
    """
    Modified from LlamaGen
    """
    def __init__(self, feature_dir, label_dir, num_examples=None):
        self.feature_dir = feature_dir
        self.label_dir = label_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        if num_examples is not None:
            self.feature_files = [f"{i}.npy" for i in range(num_examples)]
            self.label_files = [f"{i}.npy" for i in range(num_examples)]
        else:
            feature_files = {file_name for file_name in os.listdir(feature_dir) if file_name.endswith(".npy")}
            label_files = {file_name for file_name in os.listdir(label_dir) if file_name.endswith(".npy")}
            common_files = sorted(feature_files & label_files, key=_numeric_name_key)
            if not common_files:
                raise FileNotFoundError(
                    f"No matching .npy feature/label files found in {feature_dir} and {label_dir}"
                )
            self.feature_files = common_files
            self.label_files = common_files

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        aug_idx = torch.randint(low=0, high=features.shape[0], size=(1,)).item()
        features = features[aug_idx, :]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."\
        f"{feature_dir} exists: {os.path.exists(feature_dir)},"\
        f"{label_dir} exists: {os.path.exists(label_dir)}"
    return CustomDataset(feature_dir, label_dir, num_examples=getattr(args, "code_num_examples", None))


class SingleFolderDataset(Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, file_name) for file_name in os.listdir(directory)
                            if os.path.isfile(os.path.join(directory, file_name))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


def build_coco(args, transform):
    return SingleFolderDataset(args.data_path, transform=transform)


class DatasetJson(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        json_path = os.path.join(data_path, 'image_paths.json')
        assert os.path.exists(json_path), f"please first run: python3 tools/openimage_json.py"
        with open(json_path, 'r') as f:
            self.image_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                logger.warning("Failed to load image at index %s: %s", idx, e)
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')
    
    def getdata(self, idx):
        image_path = self.image_paths[idx]
        image_path_full = os.path.join(self.data_path, image_path)
        image = Image.open(image_path_full).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


class MixedDatasetJson(Dataset):
    def __init__(self, json_path, transform=None):
        super().__init__()
        self.transform = transform
        json_path = json_path
        assert os.path.exists(json_path), f"please get the json file for image path"
        with open(json_path, 'r') as f:
            self.image_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                logger.warning("Failed to load image at index %s: %s", idx, e)
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')
    
    def getdata(self, idx):
        image_path_full = self.image_paths[idx]
        image = Image.open(image_path_full).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


def build_openimage(args, transform):
    return DatasetJson(args.data_path, transform=transform)

def build_mix_img_only(args, transform):
    # the data can be distributed anywhere, and the paths are specified in the json file
    return MixedDatasetJson(args.json_path, transform=transform)

class Text2ImgDatasetImg(Dataset):
    def __init__(self, lst_dir, face_lst_dir, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl
        for lst_name in sorted(os.listdir(lst_dir)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(lst_dir, lst_name)
            valid_file_path.append(file_path)
        
        # collect valid jsonl for face
        if face_lst_dir is not None:
            for lst_name in sorted(os.listdir(face_lst_dir)):
                if not lst_name.endswith('_face.jsonl'):
                    continue
                file_path = os.path.join(face_lst_dir, lst_name)
                valid_file_path.append(file_path)            
        
        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, code_name 


class Text2ImgDataset(Dataset):
    def __init__(self, args, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl file path
        for lst_name in sorted(os.listdir(args.data_path)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(args.data_path, lst_name)
            valid_file_path.append(file_path)           
        
        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list
        self.transform = transform

        self.t5_feat_path = args.t5_feat_path
        self.short_t5_feat_path = args.short_t5_feat_path
        self.t5_feat_path_base = self.t5_feat_path.split('/')[-1]
        if self.short_t5_feat_path is not None:
            self.short_t5_feat_path_base = self.short_t5_feat_path.split('/')[-1]
        else:
            self.short_t5_feat_path_base = self.t5_feat_path_base
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

    def __len__(self):
        return len(self.img_path_list)

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        valid = 0
        return img, t5_feat_padding, attn_mask, valid

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        try:
            img = Image.open(img_path).convert("RGB")                
        except:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            return img, t5_feat_padding, attn_mask, torch.tensor(valid)

        if min(img.size) < self.image_size:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            return img, t5_feat_padding, attn_mask, torch.tensor(valid)

        if self.transform is not None:
            img = self.transform(img)
        
        t5_file = os.path.join(self.t5_feat_path, code_dir, f"{code_name}.npy")
        if torch.rand(1) < 0.3:
            t5_file = t5_file.replace(self.t5_feat_path_base, self.short_t5_feat_path_base)
        
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        if os.path.isfile(t5_file):
            try:
                t5_feat = torch.from_numpy(np.load(t5_file))
                t5_feat_len = t5_feat.shape[1] 
                feat_len = min(self.t5_feature_max_len, t5_feat_len)
                t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
                emb_mask = torch.zeros((self.t5_feature_max_len,))
                emb_mask[-feat_len:] = 1
                attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
                T = self.t5_feature_max_len
                attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
                eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
                attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
                attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
                valid = 1
            except:
                img, t5_feat_padding, attn_mask, valid = self.dummy_data()
        else:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            
        return img, t5_feat_padding, attn_mask, torch.tensor(valid)


class Text2ImgDatasetCode(Dataset):
    def __init__(self, args):
        pass




def build_t2i_image(args, transform):
    return Text2ImgDatasetImg(args.data_path, args.data_face_path, transform)

def build_t2i(args, transform):
    return Text2ImgDataset(args, transform)

def build_t2i_code(args):
    return Text2ImgDatasetCode(args)

def build_pexels(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        return build_t2i_code(args, **kwargs)
    if args.dataset == 'imagenet_openimage':
        return build_mix_img_only(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')
