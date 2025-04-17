# data.py

import itertools
import os
import logging
from PIL import Image
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler
from torch.utils.data import DataLoader

NO_LABEL = -1

# Helper function placeholder (replace with actual implementation if available)
def assert_exactly_one(args_list):
    true_args = sum(1 for arg in args_list if arg)
    assert true_args == 1, "Exactly one of the arguments must be True/provided"

# === Transformation Classes ===

class RandomTranslateWithReflect:
    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))
        new_image.paste(old_image, (xpad, ypad))
        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))
        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

# === Data Loading Helpers ===

def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    labeled_idxs_map = {} # Store original index to new label
    original_imgs = list(dataset.imgs) # Make a copy to modify safely

    for idx in range(len(original_imgs)):
        path, _ = original_imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_name = labels[filename]
            if label_name in dataset.class_to_idx:
                 label_idx = dataset.class_to_idx[label_name]
                 dataset.imgs[idx] = (path, label_idx) # Update the dataset's list
                 labeled_idxs_map[idx] = label_idx
                 del labels[filename] # Mark label as used
            # else: # Optional: Handle case where label name isn't a valid class
            #      print(f"Warning: Label '{label_name}' for file '{filename}' not in class_to_idx. Treating as unlabeled.")
            #      dataset.imgs[idx] = (path, NO_LABEL)
            #      unlabeled_idxs.append(idx)

        else:
            dataset.imgs[idx] = (path, NO_LABEL) # Update the dataset's list
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of labels contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        # Consider making this a warning instead of an error if appropriate
        logging.warning(message.format(len(labels), some_missing))
        # raise LookupError(message.format(len(labels), some_missing))


    labeled_idxs = sorted(labeled_idxs_map.keys())

    # Update dataset targets (important for SubsetRandomSampler if used later)
    dataset.targets = [label if idx not in unlabeled_idxs else NO_LABEL for idx, (_, label) in enumerate(dataset.imgs)]


    return labeled_idxs, unlabeled_idxs


# === Sampler ===

class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices # Typically unlabeled
        self.secondary_indices = secondary_indices # Typically labeled
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0, \
            f"Not enough primary indices ({len(self.primary_indices)}) for primary batch size ({self.primary_batch_size})"
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0, \
            f"Not enough secondary indices ({len(self.secondary_indices)}) for secondary batch size ({self.secondary_batch_size})"


    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            # Note: the order is changed here to unlabeled first, then labeled
            # This matches the original intent where primary=unlabeled, secondary=labeled
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        # Length based on unlabeled data iterator
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    # Use zip_longest if you need to handle the last batch potentially being smaller
    # from itertools import zip_longest
    # return zip_longest(*args, fillvalue=None)
    # Using zip drops the last incomplete batch, matching BatchSampler(drop_last=True)
    return zip(*args)


# === Dataset Configurations ===

def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 1000
    }

def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = TransformTwice(transforms.Compose([
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        # Note: Adjust 'datadir' if your CIFAR-10 structure is different
        # This path assumes images are organized in class subfolders (by-image)
        'datadir': 'mean-teacher/pytorch/data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }

# Add SVHN dataset config if needed
def svhn():
     channel_stats = dict(mean=[0.4377, 0.4438, 0.4728],
                          std=[0.1980, 0.2010, 0.1970])
     # Note: Original Mean Teacher paper didn't use HFlip for SVHN
     # RandomAffine matches the translation used in the Lightning code
     train_transformation = TransformTwice(transforms.Compose([
         # RandomTranslateWithReflect(4), # Equivalent to translate 4/32 = 0.125. Lightning code used 0.0626.
         transforms.RandomAffine(degrees=0, translate=(0.0626, 0.0626)),
         transforms.ToTensor(),
         transforms.Normalize(**channel_stats)
     ]))
     eval_transformation = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(**channel_stats)
     ])
     return {
         'train_transformation': train_transformation,
         'eval_transformation': eval_transformation,
         # Note: Adjust 'datadir' if your SVHN structure is different
         'datadir': 'data-local/images/svhn', # Example path
         'num_classes': 10
     }


# === Main Data Loader Creation Function ===

def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = relabel_dataset(dataset, labels)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader