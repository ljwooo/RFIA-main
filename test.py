import argparse
import logging

import torch

from dataset import NumpyImages
from utils import *
import csv

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--log-dir', type=str, default='results.log')
parser.add_argument('--model-name', type=str, default='')
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--output_file', type=str, default='./log.csv')

args = parser.parse_args()

args.model_name = args.model_name.split(",")

log_file_dir = args.log_dir
logging.basicConfig(filename=log_file_dir, 
                    format = '%(message)s',
                    level=logging.WARNING)

logging.warning(log_file_dir)

def evaluate(model):
    n_img, n_correct = 0, 0
    for images, names, labels in load_image(args.dir, args.image_size, args.batch_size):
        labels -= 1
        images_normlize = resnet_nomalize(images)
        img, label = torch.from_numpy(images_normlize).cuda(), torch.from_numpy(labels).cuda()
        img = img.permute(0, 3, 1, 2)
        img = torch.tensor(img, dtype=torch.float32)
        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1).view(1,-1)
        n_correct += (label != pred.squeeze(0)).sum().item()
        n_img += len(label)
    return round(100. * n_correct / n_img, 2)


adv_success_rate = []
for mname in args.model_name:
    model, data_config = build_model(mname)
    accuracy = evaluate(model)
    print(mname, accuracy)
    adv_success_rate.append(accuracy)

with open(args.output_file, 'a+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([' '])
    writer.writerow([args.dir])
    writer.writerow(args.model_name)
    writer.writerow(adv_success_rate)