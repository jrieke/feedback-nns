import torch
from torchvision import transforms, datasets
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import utils


def load_mnist(val_size=5000, seed=None):
    """Return the train (55k), val (5k, randomly drawn from the original test set) and test (10k) dataset for MNIST."""
    image_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    raw_train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=image_transform)
    test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=image_transform)

    # Split 5k samples from the train dataset for validation (similar to Sacramento et al. 2018).
    utils.seed_torch(seed)
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(raw_train_dataset, (len(raw_train_dataset)-val_size, val_size))

    return train_dataset, val_dataset, test_dataset


def load_emnist(val_size=10000, seed=None):
    """Return the train (55k), val (5k, randomly drawn from the original test set) and test (10k) dataset for MNIST."""
    image_transform = transforms.Compose([
                           # EMNIST images are flipped and rotated by default, fix this here.
                           transforms.RandomHorizontalFlip(1),
                           transforms.RandomRotation((90, 90)),

                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    target_transform = lambda x: x-1  # make labels start at 0 instead of 1

    raw_train_dataset = datasets.EMNIST('data/emnist', split='letters', train=True, download=True, transform=image_transform, target_transform=target_transform)
    test_dataset = datasets.EMNIST('data/emnist', split='letters', train=False, download=True, transform=image_transform, target_transform=target_transform)

    # Split 5k samples from the train dataset for validation (similar to Sacramento et al. 2018).
    utils.seed_torch(seed)
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(raw_train_dataset, (len(raw_train_dataset)-val_size, val_size))
    
    return train_dataset, val_dataset, test_dataset


class AddGaussianNoise():
    """An image transform that adds Gaussian noise to the image."""
    def __init__(self, mean=0, std=64, scaling_factor=0.5):
        self.mean = mean
        self.std = std
        self.scaling_factor = scaling_factor

    def __call__(self, img):
        img_array = np.asarray(img)
        noisy_img_array = img_array + self.scaling_factor * np.random.normal(self.mean, self.std, img_array.shape)
        noisy_img_array = np.clip(noisy_img_array, 0, 255)
        noisy_img_array = noisy_img_array.astype(img_array.dtype)
        return Image.fromarray(noisy_img_array)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={}, scaling_factor={})'.format(self.mean, self.std, self.intensity)


# TODO: Maybe use normal load methods from above here and exchange the transform.
def load_noisy_mnist(mean=0, std=64, scaling_factor=0.5):
    """Return the test dataset of MNIST with added Gaussian noise."""
    image_transform = transforms.Compose([
                           AddGaussianNoise(mean=mean, std=std, scaling_factor=scaling_factor),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    return datasets.MNIST('data/mnist', train=False, download=True, transform=image_transform)


def load_noisy_emnist(mean=0, std=64, scaling_factor=0.5):
    """Return the test dataset of MNIST with added Gaussian noise."""
    image_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(1),  # EMNIST images are flipped and rotated by default
                           transforms.RandomRotation((90, 90)),
                           AddGaussianNoise(mean=mean, std=std, scaling_factor=scaling_factor),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    target_transform = lambda x: x-1  # make labels start at 0 instead of 1

    return datasets.EMNIST('data/emnist', split='letters', train=False, download=True, transform=image_transform, target_transform=target_transform)


class ImageSequenceDataset(torch.utils.data.Dataset):
    """A dataset where each sample consists of a sequence of images and labels."""
    def __init__(self, allowed_seqs, image_dataset, num_classes, num_samples=10000, noisy_image_dataset=None):
        self.num_samples = num_samples
        self.allowed_seqs = allowed_seqs
        self.images_per_class = self._split_images_into_classes(image_dataset, num_classes)
        if noisy_image_dataset is None:
            self.add_noise = False
        else:
            self.add_noise = True
            self.noisy_images_per_class = self._split_images_into_classes(noisy_image_dataset, num_classes)

    def _split_images_into_classes(self, dataset, num_classes):
        images_per_class = {i: [] for i in range(num_classes)}
        for image, class_ in dataset:
            images_per_class[class_.item()].append(image)
        return images_per_class

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        seq = self.allowed_seqs[i % len(self.allowed_seqs)]
        if self.add_noise:
            images = [random.choice(self.images_per_class[class_]) for class_ in seq[:3]] + [random.choice(self.noisy_images_per_class[class_]) for class_ in seq[3:]]
        else:
            images = [random.choice(self.images_per_class[class_]) for class_ in seq]
        return torch.cat(images), seq


def plot_sequence(images, targets, target_transform=None):
    """Plot a sequence of images and corresponding labels."""
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i], cmap='Greys')
        plt.axis('off')
        plt.title(targets[i] if target_transform is None else target_transform(targets[i]))




# TODO: Delete this once the EMNIST dataset is available again.
# Temporary workaround to process the emnist dataset, because it cannot automatically be
# downloaded through pytorch (due to the shutdown).
def process_emnist():

    def get_int(b):
        return int(codecs.encode(b, 'hex'), 16)

    def read_label_file(path):
        with open(path, 'rb') as f:
            data = f.read()
            assert get_int(data[:4]) == 2049
            length = get_int(data[4:8])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
            return torch.from_numpy(parsed).view(length).long()

    def read_image_file(path):
        with open(path, 'rb') as f:
            data = f.read()
            assert get_int(data[:4]) == 2051
            length = get_int(data[4:8])
            num_rows = get_int(data[8:12])
            num_cols = get_int(data[12:16])
            images = []
            parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
            return torch.from_numpy(parsed).view(length, num_rows, num_cols)

    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')

    def _training_file(split):
        return 'training_{}.pt'.format(split)

    def _test_file(split):
        return 'test_{}.pt'.format(split)

    import os
    import codecs
    from six.moves import urllib
    import gzip
    import shutil
    import zipfile

    root = 'data/emnist'
    processed_folder = 'processed'
    raw_folder = 'data/emnist/raw'

    # process and save as torch files
    for split in splits:
        print('Processing ' + split)
        training_set = (
            read_image_file(os.path.join(raw_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
            read_label_file(os.path.join(raw_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
        )
        test_set = (
            read_image_file(os.path.join(raw_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
            read_label_file(os.path.join(raw_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
        )
        with open(os.path.join(root, processed_folder, _training_file(split)), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(root, processed_folder, _test_file(split)), 'wb') as f:
            torch.save(test_set, f)

    print('Done!')
