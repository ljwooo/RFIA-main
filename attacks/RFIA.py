import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import update_and_clip, to_np_uint8
# import tensorflow as tf
# from skimage.measure import compare_ssim as ssim

__all__ = ["RFIA"]
origin_grad = 0
origin_grad_resemble = 0
origin_grad_resemble1 = 0
origin_grad_resemble2 = 0
origin_grad_resemble3 = 0
origin_grad_resemble4 = 0
real_grad = 0
back_grads = []
ori_ilout_global = 0
ori_ilout2 = 0
tuplegrad = None

image_width = 299
image_resize = 330
prob =0.5
a = torch.tensor(0, dtype=torch.float32).cuda()
a.requires_grad_(True)
def hook_ilout(module, input, output):
    module.output = output

def get_origin_backward(module, grad_input, grad_output):
    global origin_grad
    origin_grad = grad_input[0].clone()

def get_origin_backward_aggragate(module, grad_input, grad_output):
    global origin_grad_resemble, tuplegrad
    origin_grad_resemble += grad_input[0].clone()
    tuplegrad = grad_input



def get_origin_backward_aggragate_inception_v4(module, grad_input, grad_output):
    global origin_grad_resemble1, mixer5b_gradoutput,mixer5b_gradinput, origin_grad_resemble2, origin_grad_resemble3, origin_grad_resemble4
    origin_grad_resemble1 += grad_input[0].clone()
    origin_grad_resemble2 += grad_input[1].clone()
    origin_grad_resemble3 += grad_input[2].clone()
    mixer5b_gradoutput = grad_output
    mixer5b_gradinput = grad_input

def get_origin_backward_aggragate_inception_v3(module, grad_input, grad_output):
    global origin_grad_resemble1, mixer5b_gradoutput,mixer5b_gradinput, origin_grad_resemble2, origin_grad_resemble3, origin_grad_resemble4
    origin_grad_resemble1 += grad_input[0].clone()
    origin_grad_resemble2 += grad_input[1].clone()
    origin_grad_resemble3 += grad_input[2].clone()
    origin_grad_resemble4 += grad_input[3].clone()
    mixer5b_gradoutput = grad_output
    mixer5b_gradinput = grad_input


def get_hook_pd(ori_ilout, gamma):
    def hook_pd(module, input, output):
        return gamma * output + (1-gamma) * ori_ilout
    return hook_pd

def get_hook_pd_2(ori_ilout, gamma):
    def hook_pd1(module, grad_input, grad_output):
        global real_grad, back_grads
        return gamma * grad_input[0] + (1-gamma) * ori_ilout,
    return hook_pd1



def get_hook_pd_inception_v4(ori_ilout,ori_ilout1,ori_ilout2,ori_ilout3, gamma):
    def hook_pd1(module, grad_input, grad_output):
        global real_grad, back_grads
        tuplegrad = ori_ilout, ori_ilout1, ori_ilout2
        return tuplegrad
    return hook_pd1

def get_hook_pd_inception_v3(ori_ilout,ori_ilout1,ori_ilout2,ori_ilout3, gamma):
    def hook_pd1(module, grad_input, grad_output):
        global real_grad, back_grads
        tuplegrad = ori_ilout, ori_ilout1, ori_ilout2, ori_ilout3
        return tuplegrad
    return hook_pd1


def input_diversity(input_tensor):
    rnd = torch.randint(image_width, image_resize, [1])[0]
    rescaled = F.interpolate(input_tensor, [rnd, rnd], mode='nearest')
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, [1])[0]
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, [1])[0]
    pad_right = w_rem - pad_left
    Zeropaded = nn.ZeroPad2d(padding=(pad_left, pad_right, pad_top, pad_bottom))
    d_input = Zeropaded(rescaled)
    d_input = F.interpolate(d_input, [image_width, image_width], mode='nearest')
    prob = torch.rand([1])[0]
    if prob > 0.5:
        return d_input
    else:
        return input_tensor
class RFIA(object):
    def __init__(self, args, **kwargs):
        super(RFIA, self).__init__()
        print("RFIA attacking ...")
        self.model = kwargs["source_model"]
        self.coef = args.ilpd_coef
        self.coef_forward = 0.1
        self.model_name = args.model_name
        self.approach = args.approach
        self.il_pos = args.ilpd_pos
        self.il_pos2 = args.ilpd_pos2
        print(self.il_pos, self.il_pos2)
        self.sigma = args.ilpd_sigma
        self.N = args.ilpd_N
        self.integrated_step = args.Integrated_steps
        self._select_pos()
        if 'ILPD' in self.approach:
            hook_func = get_hook_pd(0, 1)
            self.hook_forward = self.il_module.register_forward_hook(hook_func)
        if 'RFIA' in self.approach:
            hook_func2 = get_hook_pd_2(0, 1)
            self.hook_backward = self.il_module2.register_backward_hook(hook_func2)

    def __call__(self, args, ori_img, label, verbose=True):
        adv_img = ori_img.clone()
        ori_img_copy = ori_img.clone()
        ori_img_copy.requires_grad_(True)

        if 'RFIA' in self.approach:
            global origin_grad_resemble, origin_grad_resemble1, origin_grad_resemble2, origin_grad_resemble3, origin_grad_resemble4
            origin_grad_resemble = 0
            origin_grad_resemble1 = 0
            origin_grad_resemble2 = 0
            origin_grad_resemble3 = 0
            origin_grad_resemble4 = 0
            self._prep_hook_back_aggragate(ori_img_copy, 1, label, self.model_name, self.approach)
        input_grad = 0
        for i in range(args.steps):
            for j in range(self.N):
                adv_img.requires_grad_(True)
                if 'ILPD' in self.approach:
                    self._prep_hook(ori_img, j)
                logits_adv = self.model(adv_img)
                loss = F.cross_entropy(logits_adv, label)
                grad = torch.autograd.grad(loss, adv_img)[0].data
                input_grad += grad / torch.norm(grad, dim=[1,2,3], p=1, keepdim=True)
            input_grad /= self.N
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, loss {:.4f}".format(i, loss.item()))
        noise = 0
        for i in range(ori_img.size(0)):
            noise += (ori_img[i] - adv_img[i]).norm(p=2)
        print('l2norm:', noise / ori_img.size(0))
        print('infinite norm:', (ori_img - adv_img).norm(p=np.inf))
        adv_img_for_ila = adv_img.clone()
        adv_img = adv_img.permute(0,2,3,1)
        if 'RFIA' in self.approach:
            self.hook_backward.remove()
        if 'ILPD' in self.approach:
            self.hook_forward.remove()
        return to_np_uint8(adv_img), adv_img_for_ila

    def _prep_hook(self, ori_img, iteration):
        if self.sigma == 0 and iteration > 0:
            return
        self.hook_forward.remove()
        with torch.no_grad():
            ilout_hook = self.il_module.register_forward_hook(hook_ilout)
            self.model(ori_img + torch.normal(0.0, 0.1, ori_img.size()).to(ori_img.device))
            ori_ilout = self.il_module.output
            ilout_hook.remove()
        hook_func = get_hook_pd(ori_ilout, self.coef_forward)
        self.hook_forward = self.il_module.register_forward_hook(hook_func)


    def _prep_hook_back_aggragate(self, ori_img, iteration, labels, model_name, approach):
        if self.sigma == 0 and iteration > 0:
            return
        self.hook_backward.remove()
        if model_name == 'tv_resnet50' or model_name == 'tv_resnet152' or model_name == 'vgg19':
            ilout_hook = self.il_module2.register_backward_hook(get_origin_backward_aggragate)
        elif model_name == 'inception_v3':
            ilout_hook = self.il_module2.register_backward_hook(get_origin_backward_aggragate_inception_v3)
        elif model_name == 'inception_v4':
            ilout_hook = self.il_module2.register_backward_hook(get_origin_backward_aggragate_inception_v4)
        ens = self.integrated_step
        img_base = torch.zeros_like(ori_img)
        for i in range(ens):
            if approach == 'RFIA-A':
                img_noise = ori_img * (1 - i / ens) + (i / ens) * img_base
            elif approach == 'RFIA-B':
                img_noise = ori_img + torch.normal(0.0, 0.1, ori_img.size()).to(ori_img.device)
            elif approach == 'RFIA-C':
                mask = torch.from_numpy(np.random.binomial(1, 0.9, size=ori_img.size())).to(ori_img.device)
                img_noise = ori_img * mask
            elif approach == 'RFIA-AB':
                img_noise = ori_img + torch.normal(0.0, 0.1, ori_img.size()).to(ori_img.device)
                img_noise = img_noise * (1 - i / ens) + (i / ens) * img_base
            elif approach == 'RFIA-AC':
                mask = torch.from_numpy(np.random.binomial(1, 0.9, size=ori_img.size())).to(ori_img.device)
                img_noise = ori_img * mask
                img_noise = img_noise * (1 - i / ens) + (i / ens) * img_base
            elif approach == 'RFIA-BC':
                mask = torch.from_numpy(np.random.binomial(1, 0.9, size=ori_img.size())).to(ori_img.device)
                img_noise = ori_img + torch.normal(0.0, 0.1, ori_img.size()).to(ori_img.device)
                img_noise = img_noise * mask
            else:
                ValueError('no such approach!')
            logits = self.model(img_noise)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

        ilout_hook.remove()
        if model_name == 'inception_v4':
            hook_func = get_hook_pd_inception_v4(origin_grad_resemble1 / ens, origin_grad_resemble2 / ens, origin_grad_resemble3 / ens, origin_grad_resemble4 / ens, self.coef)
        elif model_name == 'inception_v3':
            hook_func = get_hook_pd_inception_v3(origin_grad_resemble1 / ens, origin_grad_resemble2 / ens, origin_grad_resemble3 / ens, origin_grad_resemble4 / ens, self.coef)
        elif model_name == 'tv_resnet50' or model_name == 'tv_resnet152' or model_name == 'vgg19':
            hook_func = get_hook_pd_2(origin_grad_resemble / ens, self.coef)
        else:
            ValueError('no such source model!')
        self.hook_backward = self.il_module2.register_backward_hook(hook_func)
    
    def _select_pos(self):
        self.model = nn.DataParallel(nn.Sequential(nn.Identity(),
                                                   self.model.module[0],
                                                   self.model.module[1]))
        if 'resnet50' in self.model_name:
            if self.il_pos == "input":
                self.il_module = eval("self.model.module[0]")
            else:
                self.il_module = eval("self.model.module[2].{}[{}]".format(*self.il_pos.split(".")))
                self.il_module2 = eval("self.model.module[2].{}[{}]".format(*self.il_pos2.split(".")))
        if 'resnet152' in self.model_name:
            if self.il_pos == "input":
                self.il_module = eval("self.model.module[0]")
            else:
                self.il_module = eval("self.model.module[2].{}[{}]".format(*self.il_pos.split(".")))
                self.il_module2 = eval("self.model.module[2].{}[{}]".format(*self.il_pos2.split(".")))
        elif 'vgg' in self.model_name:
            if self.il_pos == "input":
                self.il_module = eval("self.model.module[0]")
            else:
                self.il_module = eval("self.model.module[2].{}[{}]".format(*self.il_pos.split(".")))
                self.il_module2 = eval("self.model.module[2].{}[{}]".format(*self.il_pos2.split(".")))
        elif 'inception_v4' == self.model_name:
            if self.il_pos == "input":
                self.il_module = eval("self.model.module[0]")
            else:
                self.il_module = eval("self.model.module[2].{}[{}]".format(*self.il_pos.split(".")))
                self.il_module2 = eval("self.model.module[2].{}[{}]".format(*self.il_pos2.split(".")))
        elif 'inception_v3' == self.model_name:
            if self.il_pos == "input":
                self.il_module = eval("self.model.module[0]")
            else:
                self.il_module = eval("self.model.module[2].{}".format(self.il_pos))
                self.il_module2 = eval("self.model.module[2].{}".format(self.il_pos2))
