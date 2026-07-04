# Concept Tokenizer note: modified from 1d-tokenizer (https://github.com/bytedance/1d-tokenizer).
"""This file contains the definition of data loader using webdataset.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    https://github.com/huggingface/open-muse/blob/main/training/data.py
"""

import math
import logging
from typing import List, Union, Text
import webdataset as wds
import numpy as np
import torch
from torch.utils.data import default_collate, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.utils.data import Dataset
import linecache
import json
from PIL import Image
import random
from datasets import load_dataset
from io import BytesIO


Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)


def load_json(sample):
    sample['json'] = json.loads(sample['json'].decode('utf-8'))
    return sample


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def filter_by_res_ratio(min_res=256, min_ratio=0.5, max_ratio=2.0):
    def _f(sample):
        cfg = sample['json']
        h, w = cfg['original_height'], cfg['original_width']
        ratio = h/w
        longer_side = max(h, w)
        return ratio >= min_ratio and ratio <= max_ratio and longer_side >= min_res
    return _f


def process_recap_text(p):
    def _f(dictionary):
        if "recap_txt" in dictionary:
            if random.random() < p:
                recap_prefixes = ["The image " + v for v in ['depicts', "displays", 'showcases', 'features', 'shows']]
                # Convert input to string and strip whitespace
                text = dictionary["recap_txt"].decode("utf-8").strip()
                # Check if text starts with any of the phrases
                for phrase in recap_prefixes:
                    if text.startswith(phrase):
                        # Remove the phrase and any leading/trailing whitespace
                        text = text[len(phrase):].strip()
                        # Capitalize the first letter
                        text = text[0].upper() + text[1:] if text else ""
                        break

                dictionary["text"] = text.encode("utf-8")
        return dictionary

    return _f


def identity(x):
    return x


class ImageTransform:
    def __init__(self,
                 resize_shorter_edge: int = 256,
                 crop_size: int = 256,
                 random_crop: bool = True,
                 random_flip: bool = True,
                 normalize_mean: List[float] = [0., 0., 0.],
                 normalize_std: List[float] = [1., 1., 1.]):
        """Initializes the WebDatasetReader with specified augmentation parameters.

        Args:
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
        
        Raises:
            NotImplementedError: If the interpolation mode is not one of ["bicubic", "bilinear"].
        """
        train_transform = []
        interpolation = transforms.InterpolationMode.BICUBIC

        train_transform.append(
            transforms.Resize(resize_shorter_edge, interpolation=interpolation, antialias=True))
        if random_crop:
            train_transform.append(transforms.RandomCrop(crop_size))
        else:
            train_transform.append(transforms.CenterCrop(crop_size))
        if random_flip:
            train_transform.append(transforms.RandomHorizontalFlip())
        train_transform.append(transforms.ToTensor())
        # normalize_mean = [0, 0, 0] and normalize_std = [1, 1, 1] will normalize images into [0, 1],
        # normalize_mean = [0.5, 0.5, 0.5] and normalize_std = [0.5, 0.5, 0.5] will normalize images into [-1, 1].
        train_transform.append(transforms.Normalize(normalize_mean, normalize_std))

        self.train_transform = transforms.Compose(train_transform)
        self.eval_transform = transforms.Compose(
            [
                # Note that we always resize to crop_size during eval to ensure the results
                # can be compared against reference numbers on ImageNet etc.
                transforms.Resize(crop_size, interpolation=interpolation, antialias=True),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ]
        )
        logger.debug("train transform: %s", self.train_transform)
        logger.debug("eval transform: %s", self.eval_transform)


class SimpleImageDataset:
    def __init__(
        self,
        train_shards_path: Union[Text, List[Text]],
        eval_shards_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int = 12,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop = True,
        random_flip = True,
        normalize_mean: List[float] = [0., 0., 0.],
        normalize_std: List[float] = [1., 1., 1.],
        dataset_with_class_label: bool = True,
        dataset_with_text_label: bool = False,
        res_ratio_filtering = False,
    ):
        """Initializes the WebDatasetReader class.

        Args:
            train_shards_path: A string or list of string, path to the training data shards in webdataset format.
            eval_shards_path: A string or list of string, path to the evaluation data shards in webdataset format.
            num_train_examples: An integer, total number of training examples.
            per_gpu_batch_size: An integer, number of examples per GPU batch.
            global_batch_size: An integer, total number of examples in a batch across all GPUs.
            num_workers_per_gpu: An integer, number of workers per GPU.
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
        """
        transform = ImageTransform(
            resize_shorter_edge, crop_size, random_crop, random_flip,
            normalize_mean, normalize_std)

        if dataset_with_class_label:
            train_processing_pipeline = [
                wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
                wds.rename(
                    image="jpg;png;jpeg;webp",
                    class_id="cls",
                    handler=wds.warn_and_continue,
                    ),
                wds.map(filter_keys(set(["image", "class_id", "filename"]))),
                wds.map_dict(
                    image=transform.train_transform,
                    class_id=lambda x: int(x),
                    handler=wds.warn_and_continue,
                ),
            ]
        elif dataset_with_text_label:
            train_processing_pipeline = [
                # wds.map(load_json),
                # wds.select(filter_by_res_ratio()) if res_ratio_filtering else wds.map(identity),
                wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"]),only=["webp", "png", "jpg", "jpeg", "txt"]),
                wds.rename(
                    image="jpg;png;jpeg;webp",
                    class_id="cls",
                    text="text",
                    handler=wds.warn_and_continue,
                    ),
                wds.map(filter_keys(set(["image", "text", "class_id", "__key__"]))),
                wds.map_dict(
                    image=transform.train_transform,
                    class_id=lambda x: int(x),
                    text=lambda x: x.decode('utf-8') if isinstance(x, bytes) else x,
                    handler=wds.warn_and_continue,
                ),
            ]
        else:
            raise NotImplementedError

        if dataset_with_class_label:
            test_processing_pipeline = [
                wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
                wds.rename(
                    image="jpg;png;jpeg;webp",
                    class_id="cls",
                    handler=wds.warn_and_continue,
                    ),
                wds.map(filter_keys(set(["image", "class_id", "filename"]))),
                wds.map_dict(
                    image=transform.eval_transform,
                    class_id=lambda x: int(x),
                    handler=wds.warn_and_continue,
                ),
            ]
        else:
            test_processing_pipeline = [
                wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"]),only=["webp", "png", "jpg", "jpeg", "txt"]),
                wds.rename(
                    image="jpg;png;jpeg;webp",
                    class_id="cls",
                    text="text",
                    handler=wds.warn_and_continue,
                    ),
                wds.map(filter_keys(set(["image", "text", "class_id", "filename"]))),
                wds.map_dict(
                    image=transform.eval_transform,
                    class_id=lambda x: int(x),
                    text=lambda x: x.decode('utf-8') if isinstance(x, bytes) else x,
                    handler=wds.warn_and_continue,
                ),
            ]

        # Create train dataset and loader.
        pipeline = [
            wds.ResampledShards(train_shards_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=5000,
                        initial=1000),
            *train_processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / 
            (global_batch_size * num_workers_per_gpu))
        num_batches = num_worker_batches * num_workers_per_gpu
        num_samples = num_batches * global_batch_size

        # Each worker is iterating over the complete dataset.
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True,
        )
        # Add meta-data to dataloader instance for convenience.
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Create eval dataset and loader.
        pipeline = [
            wds.SimpleShardList(eval_shards_path),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            *test_processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=True, collation_fn=default_collate),
        ]
        self._eval_dataset = wds.DataPipeline(*pipeline)
        self._eval_dataloader = wds.WebLoader(
            self._eval_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader


class ParquetImageDataset:
    """Dataset class for loading images and text from parquet files.
    
    Similar to SimpleImageDataset but loads from parquet files instead of tar files.
    Uses HuggingFace datasets library to load parquet files.
    """
    def __init__(
        self,
        train_parquet_path: Union[Text, List[Text]],
        eval_parquet_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int = 12,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        normalize_mean: List[float] = [0., 0., 0.],
        normalize_std: List[float] = [1., 1., 1.],
        dataset_with_class_label: bool = False,
        dataset_with_text_label: bool = True,
    ):
        """Initializes the ParquetImageDataset class.

        Args:
            train_parquet_path: A string or list of strings, path pattern to training parquet files (e.g., "data/t2i/train-*.parquet").
            eval_parquet_path: A string or list of strings, path pattern to evaluation parquet files.
            num_train_examples: An integer, total number of training examples.
            per_gpu_batch_size: An integer, number of examples per GPU batch.
            global_batch_size: An integer, total number of examples in a batch across all GPUs.
            num_workers_per_gpu: An integer, number of workers per GPU.
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
            dataset_with_class_label: A boolean, whether the dataset has class labels (not supported for parquet yet).
            dataset_with_text_label: A boolean, whether the dataset has text labels (default True for parquet).
        """

        transform = ImageTransform(
            resize_shorter_edge, crop_size, random_crop, random_flip,
            normalize_mean, normalize_std)
        
        # Load parquet datasets
        logger.info("Loading training parquet files from: %s", train_parquet_path)
        train_dataset = load_dataset("parquet", data_files=train_parquet_path, split="train")
        logger.info("Loaded %s training examples", len(train_dataset))
        
        if eval_parquet_path:
            logger.info("Loading evaluation parquet files from: %s", eval_parquet_path)
            eval_dataset = load_dataset("parquet", data_files=eval_parquet_path, split="train")
            logger.info("Loaded %s evaluation examples", len(eval_dataset))
        else:
            eval_dataset = None
        
        # Create PyTorch dataset wrappers
        self._train_dataset = ParquetDatasetWrapper(
            train_dataset, transform, is_train=True, is_coco=False
        )
        self._eval_dataset = ParquetDatasetWrapper(
            eval_dataset, transform, is_train=False, is_coco=True
        ) if eval_dataset else None
        
        # Check if distributed training is enabled
        is_distributed = torch.distributed.is_initialized()
        if is_distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0
        
        # Create samplers for distributed training
        train_sampler = None
        eval_sampler = None
        if is_distributed:
            train_sampler = DistributedSampler(
                self._train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            if self._eval_dataset:
                eval_sampler = DistributedSampler(
                    self._eval_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
        
        # Calculate batches - each process only sees its portion of data
        # In distributed mode, each process handles num_train_examples / world_size examples
        # global_batch_size is already the total across all GPUs, so per-process batch size is per_gpu_batch_size
        effective_num_examples = num_train_examples // world_size if is_distributed else num_train_examples
        per_process_batch_size = per_gpu_batch_size  # Each process uses per_gpu_batch_size
        
        num_batches = math.ceil(effective_num_examples / per_process_batch_size)
        num_worker_batches = math.ceil(effective_num_examples / 
            (per_process_batch_size * num_workers_per_gpu))
        num_batches = num_worker_batches * num_workers_per_gpu
        num_samples = num_batches * per_process_batch_size
        
        # Create dataloaders
        self._train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=per_gpu_batch_size,
            shuffle=(train_sampler is None),  # Only shuffle if not using DistributedSampler
            sampler=train_sampler,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=default_collate,
            drop_last=True,
        )
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples
        
        if self._eval_dataset:
            self._eval_dataloader = DataLoader(
                self._eval_dataset,
                batch_size=per_gpu_batch_size,
                shuffle=False,
                sampler=eval_sampler,
                num_workers=num_workers_per_gpu,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=default_collate,
            )
        else:
            self._eval_dataloader = None

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader


class ParquetDatasetWrapper(Dataset):
    """Wrapper to convert HuggingFace dataset to PyTorch Dataset."""
    def __init__(self, dataset, transform: ImageTransform, is_train: bool, is_coco: bool):
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train
        self.is_coco = is_coco
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Extract image - handle different formats
        image_data = sample.get('image')
        
        if image_data is None:
            raise ValueError(f"Image not found in sample at index {idx}")
        
        # Check if it's already a PIL Image
        if isinstance(image_data, Image.Image):
            image = image_data.convert('RGB')
        # Check if it's a dict with bytes
        elif isinstance(image_data, dict):
            image_bytes = image_data.get('bytes')
            if image_bytes is None:
                raise ValueError(f"Image bytes not found in sample at index {idx}")
            try:
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to decode image at index {idx}: {e}")
        # Check if it's raw bytes
        elif isinstance(image_data, bytes):
            try:
                image = Image.open(BytesIO(image_data)).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to decode image at index {idx}: {e}")
        else:
            raise ValueError(f"Unexpected image format: {type(image_data)} at index {idx}")
        
        # Apply transform
        if self.is_train:
            image_tensor = self.transform.train_transform(image)
        else:
            image_tensor = self.transform.eval_transform(image)
        
        # Extract text from conversations
        if self.is_coco:
            text = sample.get('recaption', '')
        else:   
            conversations = sample.get('conversations', [])
            text = conversations[-1]['value'] if len(conversations) > 0 else ''
        # Process text: remove first 3 words and extract first sentence
        if text:
            text = text.strip()
            # Remove first 3 words
            words = text.split()
            if len(words) > 3:
                text = ' '.join(words[3:])
            
            # Extract first sentence (ending with period)
            period_idx = text.find('.')
            if period_idx != -1:
                text = text[:period_idx + 1].strip()
   
    
        return {
            'image': image_tensor,
            'text': text,
            '__key__': sample.get('id', str(idx)) if not self.is_coco else sample.get('image_id', str(idx))
        }
    

class PretoeknizedDataSetJSONL(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.jsonl_file = data_path
        self.num_lines = sum(1 for _ in open(self.jsonl_file))
        # Ensure the file is cached
        linecache.checkcache(self.jsonl_file)
        logger.info("Number of data: %s", self.num_lines)

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        line = linecache.getline(self.jsonl_file, idx + 1).strip()
        data = json.loads(line)
        return torch.tensor(data["class_id"]), torch.tensor(data["tokens"])
    

class PretokenizedWebDataset(SimpleImageDataset):
    def __init__ (
        self,
        train_shards_path: Union[Text, List[Text]],
        eval_shards_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop = True,
        random_flip = True,
        normalize_mean: List[float] = [0., 0., 0.],
        normalize_std: List[float] = [1., 1., 1.],
        process_recap = False,
        use_recap_prob = 0.95,
    ):
        """Initializes the PretokenizedWebDataset class.

        Text-to-image datasets are pretokenized with careful filtering (Tab. 7 in Supp.) to speed up the training
        """
        transform = ImageTransform(
            resize_shorter_edge, crop_size, random_crop, random_flip,
            normalize_mean, normalize_std)
        
        def decode_npy(x):
            arr = np.frombuffer(x, dtype=np.float16)
            ret = torch.tensor(arr)
            return ret
        
        def decode_text(x):
            ret = x.decode("utf-8")
            return ret

        train_processing_pipeline = [
            wds.rename(
                tokens="token.npy",
                text="txt",
                handler=wds.warn_and_continue,
            ),
            wds.map(process_recap_text(use_recap_prob) if process_recap else wds.map(identity)),
            wds.map(filter_keys(set(["tokens", "text", "aes_score", "__key__"]))),
            wds.map_dict(
                tokens=decode_npy,
                text=decode_text,
                handler=wds.warn_and_continue,
            ),
        ]
        
        test_processing_pipeline = [
            wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
            wds.rename(
                image="jpg;png;jpeg;webp",
                handler=wds.warn_and_continue,
            ),
            wds.map_dict(
                image=transform.eval_transform,
                handler=wds.warn_and_continue,
            ),
        ]


        # Create train dataset and loader.
        pipeline = [
            wds.ResampledShards(train_shards_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=5000,
                        initial=1000),
            *train_processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / 
            (global_batch_size * num_workers_per_gpu))
        num_batches = num_worker_batches * num_workers_per_gpu
        num_samples = num_batches * global_batch_size

        # Each worker is iterating over the complete dataset.
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True,
        )
        # Add meta-data to dataloader instance for convenience.
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Create eval dataset and loader.
        pipeline = [
            wds.SimpleShardList(eval_shards_path),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            *test_processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=True, collation_fn=default_collate),
        ]
        self._eval_dataset = wds.DataPipeline(*pipeline)
        self._eval_dataloader = wds.WebLoader(
            self._eval_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=False,
        )
