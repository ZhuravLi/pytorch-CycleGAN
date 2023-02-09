import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy.random as random
import os
import json

from data import inv_transform


class ReplayBuffer():
    """
    Buffer of generated images which is used to train discriminator.
    The replay buffer returns the newly added image with a probability of 0.5. 
    Otherwise, it sends an older generated image and replaces the older image 
    with the newly generated image.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor):
        data = data.detach()
        result = []
        for element in data:
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    result.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    result.append(element)
        return torch.stack(result)
    

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    

def save_model(disc_A, disc_B, gen_A2B, gen_B2A, epoch, checkpoint_dir):
    os.makedirs(os.path.join(checkpoint_dir, 'disc_A'), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'disc_B'), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'gen_A2B'), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'gen_B2A'), exist_ok=True)
    
    disc_A_path = os.path.join(checkpoint_dir, 'disc_A', f'disc_A_{epoch+1:0=4d}.pt') 
    disc_B_path = os.path.join(checkpoint_dir, 'disc_B', f'disc_B_{epoch+1:0=4d}.pt') 
    gen_A2B_path = os.path.join(checkpoint_dir, 'gen_A2B', f'gen_A2B_{epoch+1:0=4d}.pt') 
    gen_B2A_path = os.path.join(checkpoint_dir, 'gen_B2A', f'gen_B2A_{epoch+1:0=4d}.pt')
    
    torch.save(disc_A.state_dict(), disc_A_path)
    torch.save(disc_B.state_dict(), disc_B_path)    
    torch.save(gen_A2B.state_dict(), gen_A2B_path)    
    torch.save(gen_B2A.state_dict(), gen_B2A_path)  


def load_model_from_checkpoint(disc_A, disc_B, gen_A2B, gen_B2A, start_epoch, checkpoint_dir):
    disc_A_path = os.path.join(checkpoint_dir, 'disc_A', f'disc_A_{start_epoch:0=4d}.pt') 
    disc_B_path = os.path.join(checkpoint_dir, 'disc_B', f'disc_B_{start_epoch:0=4d}.pt') 
    gen_A2B_path = os.path.join(checkpoint_dir, 'gen_A2B', f'gen_A2B_{start_epoch:0=4d}.pt') 
    gen_B2A_path = os.path.join(checkpoint_dir, 'gen_B2A', f'gen_B2A_{start_epoch:0=4d}.pt')
    
    disc_A.load_state_dict(torch.load(disc_A_path))
    disc_B.load_state_dict(torch.load(disc_B_path))
    gen_A2B.load_state_dict(torch.load(gen_A2B_path))
    gen_B2A.load_state_dict(torch.load(gen_B2A_path))

    
def save_losses(losses_gen, losses_disc_A, losses_disc_B, epoch, checkpoint_dir):
    os.makedirs(os.path.join(checkpoint_dir, 'losses'), exist_ok=True)
    
    losses_dict = {
        'losses_gen': losses_gen, 
        'losses_disc_A': losses_disc_A,
        'losses_disc_B': losses_disc_B,
        }
    json_object = json.dumps(losses_dict)
    
    path = os.path.join(checkpoint_dir, 'losses', f'losses_{epoch+1:0=4d}.json')
    with open(path, "w") as f:
        f.write(json_object)
    
    
def load_losses_from_checkpoint(start_epoch, checkpoint_dir):
    """Function for loading losses. Returns list of losses"""
    path = os.path.join(checkpoint_dir, 'losses', f'losses_{start_epoch:0=4d}.json')
    with open(path, "r") as f:
        losses_dict = json.load(f) 
        l_gen = losses_dict['losses_gen']
        l_disc_A = losses_dict['losses_disc_A']
        l_disc_B = losses_dict['losses_disc_B']
        return l_gen, l_disc_A, l_disc_B
    

def print_parser_options(opt):
    '''Prints parser options'''
    print('\n----------------- Options ---------------')
    print('\n'.join(f'{k}\t {v}' for k, v in vars(opt).items()))
    

def save_results(real_A, real_B, fake_A, fake_B, cycled_A, cycled_B, epoch, output_dir):
    real_A = inv_transform(real_A.detach().cpu().squeeze(0))
    real_B = inv_transform(real_B.detach().cpu().squeeze(0))
    fake_A = inv_transform(fake_A.detach().cpu().squeeze(0))
    fake_B = inv_transform(fake_B.detach().cpu().squeeze(0))
    cycled_A = inv_transform(cycled_A.detach().cpu().squeeze(0))
    cycled_B = inv_transform(cycled_B.detach().cpu().squeeze(0))
    
    save_image(real_A, os.path.join(output_dir, 'B', f'{epoch+1:0=4d}_real.png'))
    save_image(fake_B, os.path.join(output_dir, 'B', f'{epoch+1:0=4d}_fake.png'))
    save_image(cycled_A, os.path.join(output_dir, 'B', f'{epoch+1:0=4d}_cycled.png'))
    save_image(real_B, os.path.join(output_dir, 'A', f'{epoch+1:0=4d}_real.png'))
    save_image(fake_A, os.path.join(output_dir, 'A', f'{epoch+1:0=4d}_fake.png'))
    save_image(cycled_B, os.path.join(output_dir, 'A', f'{epoch+1:0=4d}_cycled.png'))