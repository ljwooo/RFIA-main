import argparse
import os
import random

import numpy as np
import torch
from torch.backends import cudnn

from attacks import RFIA
from utils import build_dataset, build_model, load_image, save_image,resnet_nomalize, \
    to_categorical, ILAProjLoss, ila_forw_resnet50, to_np_uint8, ila_forw_resnet152, ila_forw_VGG19, ila_forw_inception_v3

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default=None)
parser.add_argument('--data-info-dir', type=str, default=None)
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--method', type=str, default='')
parser.add_argument('--approach', type=str, default='')
parser.add_argument('--ila_layer', type=str, default='')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--image_size', type=int, default=0)
parser.add_argument('--constraint', type=str, default="linf", choices=["linf", "l2"])
parser.add_argument('--epsilon', type=float, default=16)
parser.add_argument('--step-size', type=float, default=1.6)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--Integrated_steps', type=int, default=10)
# tv_resnet50,tv_resnet152,inception_v3,inception_v4,vgg19
parser.add_argument('--model-name', type=str, default="")
parser.add_argument('--force', default=False, action="store_true")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ilpd-N', default=1, type=int)
parser.add_argument('--ilpd-coef', default=0.1, type=float, help="1/gamma")
parser.add_argument('--ilpd-pos', default="none", type=str)
parser.add_argument('--ilpd-pos2', default="none", type=str)
parser.add_argument('--ilpd-sigma', default=0.05, type=float)

args = parser.parse_args()
if args.constraint == "linf":
    args.epsilon = args.epsilon / 255.
    args.step_size = args.step_size / 255.
print(args)

SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True
method = args.method
ila_layer = args.ila_layer
epsilon = args.epsilon
os.makedirs(args.save_dir, exist_ok=True)
model, data_config, model_no_para, args.ilpd_pos, args.ilpd_pos2, args.image_size = build_model(args.model_name)
attacker = RFIA(args, source_model = model)
for images, names, labels in load_image(args.input_dir, args.image_size, args.batch_size):
    labels -= 1
    labels_hot = to_categorical(labels, 1000)
    images_normlize = resnet_nomalize(images)
    ori_img, label = torch.from_numpy(images_normlize).cuda(), torch.from_numpy(labels_hot).cuda()
    ori_img = ori_img.permute(0, 3, 1, 2)
    ori_img = torch.tensor(ori_img, dtype=torch.float32)
    img_adv, img_adv_for_ila = attacker(args, ori_img, label, verbose=True)
    if 'ila' in args.method:
        m = 0
        momentum = 0
        attack_img = img_adv_for_ila.clone().cuda()
        img = ori_img.clone().cuda()
        with torch.no_grad():
            mid_output = ila_forw_inception_v3(model_no_para, ori_img, ila_layer)
            mid_original = torch.zeros(mid_output.size()).cuda()
            mid_original.copy_(mid_output)

            mid_output = ila_forw_inception_v3(model_no_para, attack_img, ila_layer)
            mid_attack_original = torch.zeros(mid_output.size()).cuda()
            mid_attack_original.copy_(mid_output)

        for _ in range(10):
            img.requires_grad_(True)
            mid_output = ila_forw_inception_v3(model_no_para, img, ila_layer)

            loss = ILAProjLoss()(
                mid_attack_original.detach(), mid_output, mid_original.detach(), 1.0
            )
            print(_, 'ila_loss:', loss.item())
            loss.backward()
            input_grad = img.grad.data
            if method == 'ila_fgsm':
                img = img.data + 2 * epsilon * torch.sign(input_grad)
            else:
                momentum += input_grad
                img = img.data + args.step_size * torch.sign(momentum)
            img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
            img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
            img = torch.clamp(img, min=0, max=1)
        noise = 0
        for i in range(ori_img.size(0)):
            noise += (ori_img[i] - img[i]).norm(p=2)
        print('l2norm:', noise / ori_img.size(0))
        print('infinite norm:', (ori_img - img).norm(p=np.inf))
        img_adv = to_np_uint8(img.permute(0,2,3,1))
    save_image(img_adv, names, args.save_dir)
print('images saved')
