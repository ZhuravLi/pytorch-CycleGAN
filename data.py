import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
from PIL import Image
import numpy.random as random
import os


transform = tt.Compose([
    tt.ToTensor(),
    tt.Resize((256, 256)),
    tt.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ])

inv_transform = tt.Compose([
    tt.Normalize((-1, -1, -1), (2, 2, 2))
])


class ImageDataset(Dataset):
    def __init__(self, image_dir, mode='train', unaligned=False, transform=transform):
        self.img_paths_a = get_image_paths(os.path.join(image_dir, mode+'A'))
        self.img_paths_b = get_image_paths(os.path.join(image_dir, mode+'B'))
        self.transform = transform
        self.unaligned = unaligned
    
    def __getitem__(self, index):
        img_path_a = self.img_paths_a[index % len(self.img_paths_a)]
        if self.unaligned:
            img_path_b = self.img_paths_b[random.randint(0, len(self.img_paths_b) - 1)]
        else:
            img_path_b = self.img_paths_b[index % len(self.img_paths_b)]
            
        img_a = Image.open(img_path_a)
        img_b = Image.open(img_path_b)
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
            
        return {'A': img_a, 'B': img_b}
    
    def __len__(self):
        return max(len(self.img_paths_a), len(self.img_paths_b))


def get_image_paths(image_dir):
    """
    Creates a list of image paths. Image ends with '.jpg'
    """
    image_paths = []
    for dirpath, _, filenames in os.walk(image_dir):
        for fname in filenames:
            if fname.endswith(".jpg"):
                    fpath = os.path.join(dirpath,fname)
                    image_paths.append(fpath)
    return sorted(image_paths)
    
    
def get_dataloader(dataset, device, num_workers, batch_size, shuffle=True):
    """Builds dataloader that moves data to a device."""
    def collate_fn(lst):
        imgs_a = torch.stack([elem['A'] for elem in lst])
        imgs_b = torch.stack([elem['B'] for elem in lst])
        batch = {'A': imgs_a, 'B': imgs_b}
        return batch
    
    dl = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    dl = DeviceDataLoader(dl, device)
    return dl


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for batch in self.dl: 
            yield {'A': batch['A'].to(self.device), 'B': batch['B'].to(self.device)}

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    