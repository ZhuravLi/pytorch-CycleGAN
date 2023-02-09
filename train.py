import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import itertools

from data import ImageDataset
from data import get_dataloader
from data import inv_transform
from model import Generator, Discriminator
from utils import print_parser_options
from utils import load_model_from_checkpoint, load_losses_from_checkpoint
from utils import save_model, save_losses
from utils import ReplayBuffer
from utils import init_weights
from utils import save_results
from losses import identity_loss, gan_loss, cycle_loss, discriminator_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs of training')
    parser.add_argument('--dataroot', type=str, default='datasets/monet2photo/', help='root directory of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/monet2photo_training', help='models are saved here')
    parser.add_argument('--output_dir', type=str, default='./output/training/', help='intermediate results are saved there')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    opt = parser.parse_args()

    print_parser_options(opt)
    
    # Setting the device
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you may run with --cuda")
    if not torch.cuda.is_available() and opt.cuda:
        print("WARNING: CUDA device is not available, so we run on CPU")
        opt.cuda = False
    device = torch.device('cuda') if opt.cuda else torch.device('cpu')
    
    # Dataset  
    dataset = ImageDataset(opt.dataroot, unaligned=True, mode='train')
    # Dataloader
    dataloader = get_dataloader(dataset, device, opt.n_cpu, batch_size=1)
    print('dataset was created')
    
    # Networks
    disc_A = Discriminator().to(device)
    disc_B = Discriminator().to(device)
    gen_A2B = Generator().to(device)
    gen_B2A = Generator().to(device)

    # Loading model and losses or initialize model, depending on start_epoch
    if opt.start_epoch > 0:
        load_model_from_checkpoint(disc_A, disc_B, gen_A2B, gen_B2A, \
            opt.start_epoch, opt.checkpoint_dir)
        losses = load_losses_from_checkpoint(opt.start_epoch, opt.checkpoint_dir)
        losses_gen, losses_disc_A, losses_disc_B = losses
    else:
        disc_A.apply(init_weights)
        disc_B.apply(init_weights)
        gen_A2B.apply(init_weights)
        gen_B2A.apply(init_weights)
        losses_gen = []
        losses_disc_A = []
        losses_disc_B = []

    # Initialize buffer for fake images
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Initialize optimizers
    optimizer_gen = torch.optim.Adam(itertools.chain(gen_A2B.parameters(), gen_B2A.parameters()), \
                                    lr=opt.lr, betas=(0.5, 0.999))
    optimizer_disc_A = torch.optim.Adam(disc_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_disc_B = torch.optim.Adam(disc_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    print('--------- Networks initialized ----------')

    # Create subdirectories for saving intermediate results
    dir_A = os.path.join(opt.output_dir, 'A')
    dir_B = os.path.join(opt.output_dir, 'B')
    os.makedirs(dir_A, exist_ok=True)
    os.makedirs(dir_B, exist_ok=True)
    
    
    ############### Training ################### 
    
    for epoch in range(opt.start_epoch, opt.start_epoch + opt.n_epochs):
        loss_gen_per_epoch = []
        loss_disc_A_per_epoch = []
        loss_disc_B_per_epoch = []
        
        disc_A.train()
        disc_B.train()
        gen_A2B.train()
        gen_B2A.train()
        
        for batch in tqdm(dataloader, ncols=80, desc='Processing batch'):       
            ########## Generators A2B and B2A ##########       
            optimizer_gen.zero_grad()
            
            # Set model input
            real_A = batch['A']
            real_B = batch['B']

            # Identity loss
            same_B = gen_A2B(real_B)
            identity_loss_B = identity_loss(real_B, same_B)
            same_A = gen_B2A(real_A)
            identity_loss_A = identity_loss(real_A, same_A)

            # Gan loss
            fake_B = gen_A2B(real_A)
            pred_fake_B = disc_B(fake_B)
            loss_gen_A2B = gan_loss(pred_fake_B)

            fake_A = gen_B2A(real_B)
            pred_fake_A = disc_A(fake_A)
            loss_gen_B2A = gan_loss(pred_fake_A)

            # Cycle_loss
            fake_B = gen_A2B(real_A)
            cycled_A = gen_B2A(fake_B)
            cycle_loss_A = cycle_loss(real_A, cycled_A)

            fake_A = gen_B2A(real_B)
            cycled_B = gen_A2B(fake_A)
            cycle_loss_B = cycle_loss(real_B, cycled_B)

            # Calculating total generator loss
            total_loss_gen = loss_gen_A2B + loss_gen_B2A + \
                cycle_loss_A + cycle_loss_B + identity_loss_B + identity_loss_A 

            # Gradient step
            total_loss_gen.backward()
            optimizer_gen.step()
            
            ########## Discriminator ##########           
            optimizer_disc_A.zero_grad()
            optimizer_disc_B.zero_grad()
            
            # Applying buffer
            fake_A_replaced = fake_A_buffer.push_and_pop(fake_A)
            fake_B_replaced = fake_B_buffer.push_and_pop(fake_B)

            # Discriminator loss
            pred_real_A = disc_A(real_A)
            pred_real_B = disc_B(real_B)
            pred_fake_A = disc_A(fake_A_replaced)
            pred_fake_B = disc_B(fake_B_replaced)

            loss_disc_A = discriminator_loss(pred_real_A, pred_fake_A)
            loss_disc_B = discriminator_loss(pred_real_B, pred_fake_B)
            
            # Gradient step
            loss_disc_A.backward()
            optimizer_disc_A.step()
            
            loss_disc_B.backward()
            optimizer_disc_B.step()
            
            # Losses per epoch
            loss_gen_per_epoch.append(total_loss_gen.item())
            loss_disc_A_per_epoch.append(loss_disc_A.item())
            loss_disc_B_per_epoch.append(loss_disc_B.item())

        # Record losses
        losses_gen.append(round(np.mean(loss_gen_per_epoch), 4))
        losses_disc_A.append(round(np.mean(loss_disc_A_per_epoch), 4))
        losses_disc_B.append(round(np.mean(loss_disc_B_per_epoch), 4))

        # Saving models and losses
        save_model(disc_A, disc_B, gen_A2B, gen_B2A, \
                   epoch=epoch, checkpoint_dir=opt.checkpoint_dir)
        save_losses(losses_gen, losses_disc_A, losses_disc_B, 
                    epoch=epoch, checkpoint_dir=opt.checkpoint_dir)
        
        # Log losses
        print("Epoch [{}/{}], loss_gen: {}, loss_dics_A: {}, loss_disc_B: {}".format(
            epoch+1, opt.start_epoch + opt.n_epochs, 
            losses_gen[-1], losses_disc_A[-1], losses_disc_B[-1]))
        
        # Saving intermediate results (last batch)
        with torch.no_grad():
            disc_A.eval()
            disc_B.eval()
            gen_A2B.eval()
            gen_B2A.eval()
            
            fake_B = gen_A2B(real_A)
            fake_A = gen_B2A(real_B)
            cycled_A = gen_B2A(fake_B)
            cycled_B = gen_A2B(fake_A)
            save_results(real_A, real_B, fake_A, fake_B, cycled_A, cycled_B, \
                        epoch, opt.output_dir)