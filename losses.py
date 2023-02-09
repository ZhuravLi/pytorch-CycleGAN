import torch
import torch.nn as nn


mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


def gan_loss(pred_generated):
    """Part of total generator loss for fooling discriminator"""
    loss = mse_loss(pred_generated, torch.ones_like(pred_generated))
    return loss

def cycle_loss(real_image, cycled_image):
    """
    Part of total generator loss. 
    After applying two generator we should obtain the same image
    """
    loss = l1_loss(real_image, cycled_image)
    return loss * 10

def identity_loss(real_image, same_image):
    """
    Part of generator image. 
    Applying generator to the image of target domain we get the same image
    """
    loss = l1_loss(real_image, same_image)
    return loss * 5

def discriminator_loss(pred_real, pred_generated):
    """Dicriminator loss with one-sided label smoothing"""
    real_loss = mse_loss(pred_real, torch.ones_like(pred_real).fill_(0.9))
    # One-sided label smoothing: 0.9 --> 1.0
    generated_loss = mse_loss(pred_generated, torch.zeros_like(pred_generated))
    total_loss = real_loss + generated_loss
    return total_loss * 0.5