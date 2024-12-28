#!/usr/bin/env python3

"""
nst_functions.py

Contains minimal functions to:
  - Prepare images for NST
  - Perform the style transfer optimization
  - Optionally run t-SNE and KL divergence
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

# Local imports
from Models import nst
from helper import random_crop, KLdivergence

def prepare_tensor_for_nst(img_array):
    """ 
    Normalizes and reshapes an image into the shape [1, 3, H, W].
    """
    # Expand to three identical channels
    c, h, w = 3, img_array.shape[0], img_array.shape[1]
    data_out = np.zeros([1, c, h, w])
    for i in range(c):
        data_out[0, i, :, :] = img_array
    data_out = data_out / np.max(np.abs(data_out))
    tensor_out = torch.from_numpy(data_out).float()
    if torch.cuda.is_available():
        tensor_out = tensor_out.cuda()
    return tensor_out

def run_nst(style_image_np, content_image_np,
            style_layers=['r11','r21','r31','r41','r51'],
            content_layers=['r42'],
            style_weights=[1e6/(n**2) for n in [64,128,256,512,512]],
            content_weights=[1.0],
            epochs=500, show_epoch=50):
    """
    Main style-transfer function:
     - style_image_np: Real image (NumPy)
     - content_image_np: Synthetic image (NumPy)
     - returns the final stylized image as a NumPy array
    """

    # Load VGG
    vgg = nst.VGG()
    vgg.load_state_dict(torch.load('Models/vgg_conv.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.cuda()

    # Prepare style/content images
    style_image_t = prepare_tensor_for_nst(style_image_np)
    content_image_t = prepare_tensor_for_nst(content_image_np)

    # The image we are optimizing
    opt_img = Variable(content_image_t.data.clone(), requires_grad=True)

    # Define loss layers, loss functions, and weights
    loss_layers = style_layers + content_layers
    loss_fns = ([nst.GramMSELoss()] * len(style_layers)) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_fns = [fn.cuda() for fn in loss_fns]
    weights = style_weights + content_weights

    # Compute style/content targets
    style_targets = [nst.GramMatrix()(layer).detach() for layer in vgg(style_image_t, style_layers)]
    content_targets = [c.detach() for c in vgg(content_image_t, content_layers)]
    targets = style_targets + content_targets

    # LBFGS optimizer
    optimizer = optim.LBFGS([opt_img])
    
    iteration = [0]
    while iteration[0] <= epochs:
        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[i]*loss_fns[i](out[i], targets[i]) for i in range(len(out))]
            total_loss = sum(layer_losses)
            total_loss.backward()
            iteration[0] += 1
            if iteration[0] % show_epoch == 0:
                print(f"Iteration {iteration[0]}, Loss = {total_loss.item():.4f}")
            return total_loss
        
        optimizer.step(closure)

    # Return the final stylized image
    stylized_np = opt_img.data.cpu().numpy()[0]  # shape: (3, H, W)
    return stylized_np
