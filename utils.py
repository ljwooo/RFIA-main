import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
import os
import numpy as np
from dataset import SelectedImagenet
from PIL import Image
# import joblib
import models as MODEL




def get_transforms(data_config, source=True):
    transforms = timm.data.transforms_factory.create_transform(
                        input_size = data_config['input_size'],
                        interpolation = data_config['interpolation'],
                        mean=(0,0,0),
                        std=(1,1,1),
                        crop_pct=data_config['crop_pct'] if not source else 1.,
                        tf_preprocessing=False,
                    )
    if not source:
        transforms.transforms = transforms.transforms[:-2]
    return transforms

def build_dataset(args, data_config):
    img_transform = get_transforms(data_config)
    dataset = SelectedImagenet(imagenet_val_dir=args.data_dir,
                               selected_images_csv=args.data_info_dir,
                               transform=img_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory = True, num_workers=4)
    return data_loader
    
def build_model(model_name):
    # model = eval("timm.models.{}(pretrained=True)".format(model_name))
    # model = torchvision.models.vgg16(pretrained=True)
    # model = torchvision.models.densenet121(pretrained=True)
    # model_save = "timm.models.{}(pretrained=True)".format(model_name)
    if model_name == 'inception_v4':
        model = eval("timm.models.{}(pretrained=True)".format(model_name))
        layer = "features.10"
        size = 299
    elif model_name == 'inception_v3':
        model = eval("timm.models.{}(pretrained=True)".format(model_name))
        layer = "Mixed_6b"
        size = 299
    elif model_name == 'tv_resnet152':
        model = eval("timm.models.{}(pretrained=True)".format(model_name))
        layer = "layer3.5"
        size = 224
    elif model_name == 'tv_resnet50':
        model = eval("timm.models.{}(pretrained=True)".format(model_name))
        layer = "layer2.3"
        size = 224
    elif model_name == 'vgg19':
        model = eval("timm.models.{}(pretrained=True)".format(model_name))
        layer = "features.17"
        size = 224
    else:
        ValueError('no such model!')
    # torch.save(model.state_dict(), 'vgg19_ILPD.pth')
    # print(model)
    # print(model.features[17])
    data_config = None
    if 'inception_v' in model_name:
        model = nn.Sequential(T.Normalize([0.5, 0.5, 0.5],
                                          [0.5, 0.5, 0.5]),
                              model)
    else:
        model = nn.Sequential(T.Normalize([0.485, 0.456, 0.408],
                                          [0.229, 0.224, 0.225]),
                              model)

    model_parallel = nn.DataParallel(model)
    model.eval()
    model.cuda()
    model_parallel.eval()
    model_parallel.cuda()
    return model_parallel, data_config, model, layer, layer, size

def resnet_nomalize(x):
    return x/255.

with open('./labels.txt') as f:
    ground_truth=f.read().split('\n')[:-1]
# ...
def load_image(image_path, image_size, batch_size):
    images = []
    filenames = []
    labels = []
    idx = 0

    files = os.listdir(image_path)
    files.sort(key=lambda x: int(x[:-4]))
    for i, filename in enumerate(files):
        # image = imread(image_path + filename)
        # image = imresize(image, (image_size, image_size)).astype(np.float)
        image = Image.open(image_path + filename)
        image = image.resize((image_size, image_size))
        image = np.array(image)
        images.append(image)
        filenames.append(filename)

        labels.append(int(ground_truth[int(files[i][:-4]) - 1]))
        # labels.append(20)
        idx += 1
        if idx == batch_size:
            yield np.array(images), np.array(filenames), np.array(labels)
            idx = 0
            images = []
            filenames = []
            labels = []
    if idx > 0:
        yield np.array(images), np.array(filenames), np.array(labels)

def save_image(images,names,output_dir):
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        # imsave(output_dir+name,images[i].astype('uint8'))
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def ila_forw_inception_v3(model, x, ila_layer):
    x = model[0](x)
    x = model[1].Conv2d_1a_3x3(x)
    x = model[1].Conv2d_2a_3x3(x)
    x = model[1].Conv2d_2b_3x3(x)
    x = model[1].Conv2d_3b_1x1(x)
    x = model[1].Conv2d_4a_3x3(x)
    x = model[1].Mixed_5b(x)
    if ila_layer == 'Mixed_5b':
        return x
    x = model[1].Mixed_5c(x)
    if ila_layer == 'Mixed_5c':
        return x
    x = model[1].Mixed_5d(x)
    if ila_layer == 'Mixed_5d':
        return x
    x = model[1].Mixed_6a(x)
    if ila_layer == 'Mixed_6a':
        return x
    x = model[1].Mixed_6b(x)
    if ila_layer == 'Mixed_6b':
        return x
    return False
def ila_forw_resnet50(model, x, ila_layer):
    jj = int(ila_layer.split('_')[0])
    kk = int(ila_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].act1(x)
    if jj == 0 and kk ==0:
        return x
    x = model[1].maxpool(x)

    for ind, mm in enumerate(model[1].layer1):
        x = mm(x)
        if jj == 1 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer2):
        x = mm(x)
        if jj == 2 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer3):
        x = mm(x)
        if jj == 3 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer4):
        x = mm(x)
        if jj == 4 and ind == kk:
            return x
    return False
def ila_forw_resnet152(model, x, ila_layer):
    jj = int(ila_layer.split('_')[0])
    kk = int(ila_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    if jj == 0 and kk ==0:
        return x
    x = model[1].maxpool(x)

    for ind, mm in enumerate(model[1].layer1):
        x = mm(x)
        if jj == 1 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer2):
        x = mm(x)
        if jj == 2 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer3):
        x = mm(x)
        if jj == 3 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer4):
        x = mm(x)
        if jj == 4 and ind == kk:
            return x
    return False

def ila_forw_VGG19(model, x, ila_layer):
    jj = ila_layer.split('.')[0]
    kk = ila_layer.split('.')[1]
    x = model[0](x)
    x = model[1].features[0](x)
    x = model[1].features[1](x)
    x = model[1].features[2](x)
    x = model[1].features[3](x)
    x = model[1].features[4](x)
    x = model[1].features[5](x)
    x = model[1].features[6](x)
    x = model[1].features[7](x)
    x = model[1].features[8](x)
    x = model[1].features[9](x)
    x = model[1].features[10](x)
    x = model[1].features[12](x)
    x = model[1].features[13](x)
    x = model[1].features[14](x)
    x = model[1].features[15](x)
    x = model[1].features[16](x)
    x = model[1].features[17](x)
    if kk == '17':
        return x
    x = model[1].features[18](x)

    return False
def to_np_uint8(x):
    return torch.round(x.data*255).cpu().numpy().astype(np.uint8())

class ILAProjLoss(torch.nn.Module):
    def __init__(self):
        super(ILAProjLoss, self).__init__()
    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        n = old_attack_mid.shape[0]
        x = (old_attack_mid - original_mid).view(n, -1)
        y = (new_mid - original_mid).contiguous().view(n, -1)
        # x_norm = x / torch.norm(x, dim = 1, keepdim = True)
        proj_loss =torch.sum(y * x) / n
        return proj_loss