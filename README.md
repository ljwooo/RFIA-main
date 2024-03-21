# RFIA-attack
Code for our paper [Towards A Simple, Strong, and Solid Baseline for Boosting Adversarial Transferability: The Devil is in the Details]().

## Requirements
* Python 3.8.8
* PyTorch 1.12.0
* Torchvision 0.13.0
* timm 0.6.11
  
## Datasets
Select images from ILSVRC 2012 validation set.


## Attack and Evaluate
### Attack
Perform attack:```attack.py``` is the entry to execute the attack methods proposed in this work.

For ``` --model-name ```, use the model name in [timm](https://github.com/huggingface/pytorch-image-models). For instance, ```  tv_resnet50 ```.
### Evaluate
Evaluate the success rate of adversarial examples:```test.py``` is to test the crafted adversarial examples for attack success rates against all involved models.

For ``` --model-name ```, use the model name in [timm](https://github.com/huggingface/pytorch-image-models). Separate different victim models using commas. For instance, ``` vit_base_patch16_224 ```.

## Acknowledgements
Code refer to [Improving Adversarial Transferability via Intermediate-level Perturbation Decay](https://arxiv.org/abs/2304.13410)

* [timm](https://github.com/huggingface/pytorch-image-models)


