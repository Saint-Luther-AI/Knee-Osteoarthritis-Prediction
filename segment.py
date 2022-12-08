import argparse
import os
from solver import Solver
from torch.backends import cudnn
import random
from torch.utils import data
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=224, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        # GT : Ground Truth
        self.GT_paths = os.path.join(root, "label")
        self.root = os.path.join(root, "image")
        self.image_paths = list(os.listdir(self.root))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = os.path.join(self.root, self.image_paths[index])
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(image, axis=0)
        GT_path = os.path.join(self.GT_paths, self.image_paths[index])
        gt = cv2.imread(GT_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (224, 224), interpolation=cv2.INTER_AREA)
        labels = np.unique(gt)
        mask = np.zeros([4, 224, 224])
        for label in labels:
            mask[label, :, :] = (gt == label).astype(np.uint8)
        GT = mask

        image = torch.from_numpy(image).type(torch.FloatTensor)
        GT = torch.from_numpy(GT).type(torch.FloatTensor)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

def get_accuracy(SR,GT):
    SR = F.softmax(SR, dim=1)
    SR = torch.argmax(SR, dim=1).squeeze(1)
    #GT = torch.argmax(GT, dim=1).squeeze(1)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)
    acc = float(corr)/float(tensor_size)
    return acc

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    lr = 0.0002
    augmentation_prob = config.augmentation_prob
    epoch = config.num_epochs
    decay_ratio = random.random() * 0.8
    decay_epoch = int(epoch * decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)

    solver = Solver(config, train_loader)

    solver.train()
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.2)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--train_path', type=str, default='data/train/')
    parser.add_argument('--valid_path', type=str, default='data/valid/')
    parser.add_argument('--test_path', type=str, default='data/test/')
    parser.add_argument('--result_path', type=str, default='result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
