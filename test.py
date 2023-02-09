import os
import shutil
import argparse

import torch
from torchvision.utils import save_image

from data import ImageDataset
from data import get_dataloader
from data import inv_transform
from model import Generator
from utils import print_parser_options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./datasets/monet2photo/', help='root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='./output/testing/', help='results are saved there')
    parser.add_argument('--num_test', type=int, default=10, help='number of images processed, in one direction')
    parser.add_argument('--shuffle', action='store_true', help='use photos in random order')
    parser.add_argument('--generator_A2B', type=str, default='./checkpoints/monet2photo_pretrained/gen_A2B.pt', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='./checkpoints/monet2photo_pretrained/gen_B2A.pt', help='B2A generator checkpoint file')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    opt = parser.parse_args()

    print_parser_options(opt)

    # Setting the device
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
    if not torch.cuda.is_available() and opt.cuda:
        print("WARNING: CUDA device is not available, so we run on CPU")
        opt.cuda = False

    device = torch.device('cuda') if opt.cuda else torch.device('cpu')

    # Dataset  
    dataset = ImageDataset(opt.dataroot, unaligned=False, mode='test')
    # Dataloader
    dataloader = get_dataloader(dataset, device, num_workers=0, batch_size=1, shuffle=opt.shuffle)
    print('dataset was created')

    # Networks
    gen_A2B = Generator().to(device)
    gen_B2A = Generator().to(device)

    gen_A2B.load_state_dict(torch.load(opt.generator_A2B))
    gen_B2A.load_state_dict(torch.load(opt.generator_B2A))

    gen_A2B.eval()
    gen_B2A.eval()
    print('--------- Networks initialized ----------')

    # Clear output_dir if it exists
    if os.path.exists(opt.output_dir):
        for root, dirs, files in os.walk(opt.output_dir):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    # Create subdirectories for saving results
    dir_A = os.path.join(opt.output_dir, 'A')
    dir_B = os.path.join(opt.output_dir, 'B')
    os.makedirs(dir_A)
    os.makedirs(dir_B)


    ############### Testing ################### 
        
    for i, batch in enumerate(dataloader):
        if i >= opt.num_test:
            break
        
        with torch.no_grad():
            # Set model input
            real_A = batch['A']
            real_B = batch['B']

            # Generate output
            fake_A = gen_B2A(real_B)
            fake_B = gen_A2B(real_A)
            
            # Prepare images for saving ([-1, -1] --> [0, 1])
            real_A = inv_transform(real_A.detach().cpu().squeeze(0))
            real_B = inv_transform(real_B.detach().cpu().squeeze(0))
            fake_A = inv_transform(fake_A.detach().cpu().squeeze(0))
            fake_B = inv_transform(fake_B.detach().cpu().squeeze(0))
            
            # Save image files
            save_image(real_A, os.path.join(dir_B, f'{i+1:0=4d}_real.png'))
            save_image(fake_B, os.path.join(dir_B, f'{i+1:0=4d}_fake.png'))
            
            save_image(real_B, os.path.join(dir_A, f'{i+1:0=4d}_real.png'))
            save_image(fake_A, os.path.join(dir_A, f'{i+1:0=4d}_fake.png'))

            print(f'Generated images {i+1:0=4d} of {opt.num_test:0=4d}')
