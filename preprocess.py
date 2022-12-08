import json
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from network import U_Net

def parse_json():  # convert the json files into segmentation masks

    files = [file for file in os.listdir('F:/Segmentation/Knee_joint/mask_json') if file.endswith(".json")]
    for name in files:
        file_path = os.path.join('F:/Segmentation/Knee_joint/mask_json', name)
        filename = os.path.splitext(name)[0]
        with open(file_path, 'r') as jsonfile:
            json_string = json.load(jsonfile)
        list = json_string['shapes'][0]
        points_0 = np.array(list['points']).astype(int)
        list = json_string['shapes'][1]
        points_1 = np.array(list['points']).astype(int)
        list = json_string['shapes'][2]
        points_2 = np.array(list['points']).astype(int)

        image_name = filename + '.png'
        image_path = os.path.join('F:/Segmentation/Knee_joint/images', image_name)
        image = cv2.imread(image_path)
        height, width = image.shape[0], image.shape[1]
        maskImage = np.zeros((height, width), dtype=np.uint8)
        for i in range(width):
            for j in range(height):
                if cv2.pointPolygonTest(points_0, (i, j), False) >= 0:
                    maskImage[j, i] = 0
                elif cv2.pointPolygonTest(points_1, (i, j), False) >= 0:
                    maskImage[j, i] = 1
                elif cv2.pointPolygonTest(points_2, (i, j), False) >= 0:
                    maskImage[j, i] = 2
                else:
                    maskImage[j, i] = 3
        cv2.imwrite(f'F:/Segmentation/Knee_joint/mask/{filename}.png', maskImage)

def generate_mask():

    unet = U_Net(img_ch=1, output_ch=4)
    unet.load_state_dict(torch.load(f'models/U_Net-50-0.0002-25-0.2000.pkl', map_location=torch.device('cpu')))
    unet.eval()

    main_dir = 'data/kneeKL224/train'
    for i in range(5):
        files = [file for file in os.listdir(f'{main_dir}/{i}')]
        for name in files:
            file_path = os.path.join(f'{main_dir}/{i}', name)
            image = cv2.imread(file_path, 0)
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
            image = torch.from_numpy(image).type(torch.FloatTensor)
            SR = unet(image)
            SR = F.softmax(SR, dim=1)
            SR = torch.argmax(SR, dim=1).squeeze(1)
            SR = np.array(SR)
            cv2.imwrite(f'{main_dir}_mask/{i}/{name}', SR[0])

    main_dir = 'data/kneeKL224/val'
    for i in range(5):
        files = [file for file in os.listdir(f'{main_dir}/{i}')]
        for name in files:
            file_path = os.path.join(f'{main_dir}/{i}', name)
            image = cv2.imread(file_path, 0)
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
            image = torch.from_numpy(image).type(torch.FloatTensor)
            SR = unet(image)
            SR = F.softmax(SR, dim=1)
            SR = torch.argmax(SR, dim=1).squeeze(1)
            SR = np.array(SR)
            cv2.imwrite(f'{main_dir}_mask/{i}/{name}', SR[0])

    main_dir = 'data/kneeKL224/test'
    for i in range(5):
        files = [file for file in os.listdir(f'{main_dir}/{i}')]
        for name in files:
            file_path = os.path.join(f'{main_dir}/{i}', name)
            image = cv2.imread(file_path, 0)
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
            image = torch.from_numpy(image).type(torch.FloatTensor)
            SR = unet(image)
            SR = F.softmax(SR, dim=1)
            SR = torch.argmax(SR, dim=1).squeeze(1)
            SR = np.array(SR)
            cv2.imwrite(f'{main_dir}_mask/{i}/{name}', SR[0])

if __name__ == '__main__':

    #parse_json()
    generate_mask()


