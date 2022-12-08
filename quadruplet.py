import argparse
import ssl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import cv2

#gets the knee joints for classification
class Knee_Classification(Dataset):
    def __init__(self, main_dir, num_class, transform=None):
        self.path = main_dir
        self.transform = transform
        self.num_class = num_class
        self.images_0, self.images_1, self.images_2, self.images_3, self.targets = [], [], [], [], []
        self.getImageLabels()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, indx):
        return self.images_0[indx], self.images_1[indx], self.images_2[indx], self.images_3[indx], self.targets[indx]

    def getImageLabels(self):

        for i in range(5):
            files = [file for file in os.listdir(f'{self.path}/{i}')]
            for name in files:
                file_path = os.path.join(f'{self.path}/{i}', name)
                image = cv2.imread(file_path)
                mask = cv2.imread(os.path.join(f'{self.path}_mask/{i}', name))
                j = 0
                for images in (self.images_0, self.images_1, self.images_2, self.images_3):
                    img = image * (mask == j)
                    img = Image.fromarray(img.astype(np.uint8))
                    if self.transform is not None:
                        img = self.transform(img)
                    images.append(img)
                    j = j + 1
                self.targets.append(torch.tensor(i, dtype=torch.int64))

if __name__ == '__main__':

    parse=argparse.ArgumentParser()
    parse.add_argument("-root_path",type=str,default="data/kneeKL224",help="root path")
    parse.add_argument("-epochs",type=int,default=12,help="the number of epochs")
    parse.add_argument("-lr", type=float, default=5e-5, help="the learning rate")
    parse.add_argument("-batch_size",type=int,default=1,help="the number of batch_size")
    parse.add_argument("-seed",type=int,default=0,help="random seed")
    parse.add_argument("-net_type", type=str, default="vgg", help="net type")
    parse.add_argument("-num_class", type=int, default=5, help="the number of classes")
    args=parse.parse_args()

    # cancer the verification of http protocol
    ssl._create_default_https_context = ssl._create_unverified_context
    # fix the seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = 'cpu'

    #create a folder to save the trained model
    models_folder = 'models'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    #load the model
    #pre_trained: vgg or resnet
    if args.net_type=="vgg":
        model=models.vgg19(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        #feature_model.append(nn.Linear(num_ftrs, args.num_class))
        model.classifier = nn.Sequential(*feature_model)

    #load the optimiser
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.0005)
    # adjust the learning_rate as  training
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=3, gamma=0.1)
    # load the loss function
    loss_function = nn.CrossEntropyLoss()

    #load the DataLoader
    pixel_mean=0.66133188
    pixel_std=0.21229856
    transform_train = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ])
    transform_val=transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)
        ])

    knee_detection_dataset_train = Knee_Classification(f'{args.root_path}/train', args.num_class, transform=transform_train)
    knee_detection_dataset_val = Knee_Classification(f'{args.root_path}/val', args.num_class, transform=transform_val)
    train_dataloader = DataLoader(knee_detection_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(knee_detection_dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print("-------------start training----------------")
    model.to(device)
    best_acc=0
    best_epoch=0
    for epoch in range(args.epochs):
        model.train()
        print("epoch:",epoch)
        for images_0, images_1, images_2,images_3, labels in train_dataloader:
            images_0, images_1, images_2,images_3, labels = images_0.to(device), images_1.to(device), images_2.to(device), images_3.to(device), labels.to(device)
            num_ftrs_0, num_ftrs_1, num_ftrs_2, num_ftrs_3 = model(images_0), model(images_1), model(images_2), model(images_3)
            num_ftrs = (torch.cat((num_ftrs_0, num_ftrs_1, num_ftrs_2, num_ftrs_3), 1)).to(device)
            FC = (nn.Linear(num_ftrs.shape[1], args.num_class)).to(device)
            logits = FC(num_ftrs)
            loss = loss_function(logits, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        lr_scheduler.step()

        # start verification
        true_labels = []
        predicted_labels = []
        if epoch%2==0:
            model.eval()
            with torch.no_grad():
                for images_0, images_1, images_2, images_3, labels in val_dataloader:
                    images_0, images_1, images_2, images_3, labels = images_0.to(device), images_1.to(device), images_2.to(device), images_3.to(device), labels.to(device)
                    num_ftrs_0, num_ftrs_1, num_ftrs_2, num_ftrs_3 = model(images_0), model(images_1), model(images_2), model(images_3)
                    num_ftrs = (torch.cat((num_ftrs_0, num_ftrs_1, num_ftrs_2, num_ftrs_3), 1)).to(device)
                    FC = (nn.Linear(num_ftrs.shape[1], args.num_class)).to(device)
                    logits = FC(num_ftrs)
                    loss = loss_function(logits, labels)

                    True_label = labels.cpu().data.numpy().flatten()
                    predictions = torch.max(logits, 1)[1].cpu().data.numpy().flatten()
                    true_labels.extend(True_label)
                    predicted_labels.extend(predictions)

            val_acc = metrics.accuracy_score(true_labels, predicted_labels)

            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                print('best acc:', best_acc, 'best epoch:', best_epoch)
                torch.save(model.state_dict(), f'{models_folder}/Adam5e-5_best_epoch_model.pth')

    print("-------------complete training----------------")

    print("-------------test starting----------------")
    model.load_state_dict(torch.load(f'{models_folder}/Adam5e-5_best_epoch_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # load the DataLoader
    transform_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([pixel_mean] * 3, [pixel_std] * 3)
    ])
    knee_detection_dataset_test = Knee_Classification(f'{args.root_path}/test', args.num_class, transform=transform_test)
    test_dataloader = DataLoader(knee_detection_dataset_test, batch_size=args.batch_size, shuffle=True)

    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images_0, images_1, images_2, images_3, labels in test_dataloader:
            images_0, images_1, images_2, images_3, labels = images_0.to(device), images_1.to(device), images_2.to(device), images_3.to(device), labels.to(device)
            num_ftrs_0, num_ftrs_1, num_ftrs_2, num_ftrs_3 = model(images_0), model(images_1), model(images_2), model(images_3)
            num_ftrs = torch.cat((num_ftrs_0, num_ftrs_1, num_ftrs_2, num_ftrs_3), 1).to(device)
            num_ftrs = num_ftrs.to(device)
            FC = nn.Linear(num_ftrs.shape[1], args.num_class)
            FC = FC.to(device)
            logits = FC(num_ftrs)
            loss = loss_function(logits, labels)

            True_label = labels.cpu().data.numpy().flatten()
            predictions = torch.max(logits, 1)[1].cpu().data.numpy().flatten()
            true_labels.extend(True_label)
            predicted_labels.extend(predictions)
    if args.num_class==5:
        targets = ['0', '1', '2', '3', '4']
        n_classes = 5
    if args.num_class==4:
        targets = ['01', '2', '3', '4']
        n_classes = 4
    if args.num_class==2:
        targets = ['0', '1']
        n_classes = 2
    print(metrics.classification_report(true_labels, predicted_labels, target_names=targets, digits=4))
    print("MAE:", metrics.mean_absolute_error(true_labels, predicted_labels))
    print("MSE:", metrics.mean_squared_error(true_labels, predicted_labels))
    ax = plt.axes()  # use the Axes object to generate the picture
    # seaborn.heatmap() function: generate the heat map
    sn.heatmap(metrics.confusion_matrix(true_labels, predicted_labels, labels=range(n_classes)), xticklabels=targets, yticklabels=targets, annot=True, fmt='g', ax=ax)
    ax.set_title('General Confusion Matrix', weight='bold')
    ax.set_xlabel("Predicted Labels", weight='bold')
    ax.set_ylabel("True Labels", weight='bold')
    plt.savefig(f"pictures/all_Manual_Confusion Matrix_Class{args.num_class}.jpg")
    plt.show()
    plt.clf()

    print("-------------complete testing----------------")
