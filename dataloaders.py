from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import torch
from utils import get_transformation
from tqdm import tqdm
import math
from multiprocessing import Pool
import torch.multiprocessing

class ImageDataset(Dataset):
    def __init__(self, files, image_size=1024, project_dir='', input_same_as_ref=False):
        self.files = files
        self.project_dir = project_dir
        self.transform = get_transformation(image_size)
        self.input_same_as_ref = input_same_as_ref

    def __len__(self):
        return len(self.files)
    
    def is_image(self, file_name):
        return file_name.split('.')[-1] in ['png', 'jpeg', 'jpg']

    def __getitem__(self, idx):
        if self.is_image(self.files[idx]):
            item = {
                'input': self.transform(Image.open(os.path.join(self.project_dir, self.files[idx])).convert("RGB"))
            }
            if self.input_same_as_ref:
                item['ref'] = item['input']
        else:
            contents = torch.load(os.path.join(self.project_dir, self.files[idx]))

            item = {
                'input': self.transform(Image.fromarray(contents['input']).convert('RGB'))
            }
            if 'ref' in contents:
                item['ref'] =  self.transform(Image.fromarray(contents['ref']).convert('RGB'))
            
            if 'mask' in contents:
                item['mask'] = contents['mask']
        return item                


class IntermediateDataset(Dataset):
    def __init__(self, files_location, 
                 verbose=False, 
                 pattern='*latents.pt',
                 gen_min=None,
                 gen_max=None,
                 latent_min=None,
                 latent_max=None,
                 only_latent=False,
                 only_gen=False,
                 bool_normalize=True):
        '''
            Args:
                files_location: path to folder
                pattern: extension of files we are interested in the folder specified by files_location
                gen_min, gen_max, latent_min, latent_max: values that are used to normalize our data
                only_latent: if True, the dataloader will return only the latent
                only_gen: if True, the dataloader will return only the intermediate output
                bool_normalize: if True, will normalize.
        '''
        self.bool_normalize = bool_normalize
        self.verbose = verbose
        self.only_latent = only_latent
        self.only_gen = only_gen
        self.files = sorted([x for x in glob.iglob(os.path.join(files_location, pattern))])
        if not self.files:
            raise Exception('Empty dataset...')
        if bool_normalize and (gen_min is None or gen_max is None or latent_min is None or latent_max is None):
            gen_min, gen_max, latent_min, latent_max = self.estimate_stats()

        self.gen_min, self.gen_max, self.latent_min, self.latent_max = gen_min, gen_max, latent_min, latent_max
        if verbose and bool_normalize: 
            print(self.gen_min, self.gen_max, self.latent_min, self.latent_max)
    
    def __len__(self):
        return len(self.files)
    
    def estimate_stats(self):
        gen_min = []
        gen_max = []
        latent_min = []
        latent_max = []
        for i in tqdm(range(len(self.files)), disable=not self.verbose):
            latents = torch.load(self.files[i], map_location='cpu')
            latent_z = latents['latent_z'][0]
            gen_out = latents['gen_outs'][-1][0] 
            gen_min.append(gen_out.min())
            gen_max.append(gen_out.max())
            latent_min.append(latent_z.min())
            latent_max.append(latent_z.max()) 
        return torch.tensor(gen_min).mean(), torch.tensor(gen_max).mean(), torch.tensor(latent_min).mean(), torch.tensor(latent_max).mean()

    def normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-8)
    

    def __getitem__(self, idx):
        latents = torch.load(self.files[idx], map_location='cpu')
        latent_z = latents['latent_z'].squeeze()
        gen_out = latents['gen_outs'][-1].squeeze()

        latent_z = latent_z.permute(1, 0)
        if self.bool_normalize:
            latent_z = self.normalize(latent_z, self.latent_min, self.latent_max)
        img_dim = gen_out.shape[-1]
        gen_out = gen_out.reshape(gen_out.shape[0], -1)

        if self.bool_normalize:
            gen_out = self.normalize(gen_out, self.gen_min, self.gen_max)
        if self.only_latent:
            return latent_z.unsqueeze(-1)
        if self.only_gen:
            return gen_out.reshape(-1, img_dim, img_dim)
        else:
            # add some zeros to create a square image
            added_zeros = int((img_dim + 1) ** 2 - (img_dim ** 2 + latent_z.shape[1]))
            concatenated = torch.cat([gen_out.detach(), latent_z.detach(), torch.zeros(gen_out.shape[0], added_zeros)], 1).reshape(gen_out.shape[0], img_dim + 1, img_dim + 1)
            return concatenated
    
    def estimate_latents_stats(self, num_latents=3000):
        with torch.no_grad():
            latents = [self.__getitem__(x)[:, :, 0] for x in range(num_latents)]
        latents = torch.stack(latents, -1).permute(1, 0, 2).to('cuda')
        mean_latents = latents - latents.mean(dim=-1).unsqueeze(-1)
        cov = (mean_latents @ mean_latents.permute(0, 2, 1)) / (num_latents - 1)
        
        norm = torch.linalg.norm(cov - torch.eye(512).to('cuda')) / 18
        print(f'Mean: {latents.mean()}, Std: {latents.std()}')
        print(f'Covariance Frobenius norm: {norm}')

    

if __name__ == '__main__':
    dataset = IntermediateDataset('../datasets/inversions/FFHQ_0_0_0_300/', bool_normalize=False, only_latent=True)
    dataset.estimate_latents_stats()
    # dataset = IntermediateDataset('../datasets/CelebaInvertedv2/', verbose=True)
