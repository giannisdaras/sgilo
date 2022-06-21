import torch
import sys
import numpy as np
from torch import nn
from torch.nn import functional as F
from PIL import Image
import os
from torchvision import transforms
import shutil
from collections import OrderedDict
import math
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision
from datetime import datetime
import glob
import torch.distributed as dist
import functools
import lpips
import logging
import warnings
import pickle
import copy
from score.models import ViT
import score.sde as sde_lib
from score.sde import *
from torch.optim import Adam
try:
    import clip
except:
    warnings.warn('Clip not found... Attempting to continue without it')
import yaml



class SphericalOptimizer():
    def __init__(self, params):
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True)+1e-9).sqrt() for param in params}
    @torch.no_grad()
    def step(self, closure=None):
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt())
            param.mul_(self.radii[param])


class MappingProxy(nn.Module):
    def __init__(self, gaussian_ft):
        super(MappingProxy, self).__init__()
        self.mean = torch.nn.Parameter(gaussian_ft["mean"], requires_grad=False)
        self.std = torch.nn.Parameter(gaussian_ft["std"], requires_grad=False)
        self.lrelu = torch.nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.lrelu(self.std * x + self.mean)
        return x


class BicubicDownSample(nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.:
            return (a + 2.) * torch.pow(abs_x, 3.) - (a + 3.) * torch.pow(abs_x, 2.) + 1
        elif 1. < abs_x < 2.:
            return a * torch.pow(abs_x, 3) - 5. * a * torch.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
        else:
            return 0.0

    def __init__(self, factor=4, device='cuda', padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
                          for i in range(size)], dtype=torch.float32)
        k = k / torch.sum(k)
        # k = torch.einsum('i,j->ij', (k, k))
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0).float().to(device)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0).float().to(device)
        self.device = device
        self.padding = padding
        #self.padding = 'constant'
        #self.padding = 'replicate'
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor
        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters1 = self.k1
        filters2 = self.k2
        # filters1 = self.k1.type('torch{}.FloatTensor'.format(self.device))
        # filters2 = self.k2.type('torch{}.FloatTensor'.format(self.device))
        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(
                x, 2, 3), 1, 2)   # NHWC to NCHW
        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x.float(), weight=filters1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)
        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)
        if nhwc:
            x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        if byte_output:
            return x.type('torch.ByteTensor'.format(self.cuda))
        else:
            return x


def loss_geocross(latent):
    if latent.size() == (1, 512):
        return 0
    else:
        num_latents  = latent.size()[1]
        X = latent.view(-1, 1, num_latents, 512)
        Y = latent.view(-1, num_latents, 1, 512)
        A = ((X - Y).pow(2).sum(-1) + 1e-12).sqrt()
        B = ((X + Y).pow(2).sum(-1) + 1e-12).sqrt()
        D = 2 * torch.atan2(A, B)
        D = ((D.pow(2) * 512).mean((1, 2)) / 8.).mean()
        return D


def get_lr(t, initial_lr, rampdown=0.75, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)


def project_onto_l1_ball(x, eps):
    """
    See: https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


def project_onto_l1_ball(x, eps):
    """
    See: https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


# Source: https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def pretty(d, indent=0):
    ''' Print dictionary '''
    for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def get_transformation(image_size):
    return transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def load_classifier_from_robustness(cls, cls_path):
    state_dict = torch.load(cls_path)['model']
    new_dict = OrderedDict()
    for key in state_dict.keys():
        if 'module.model' in key:
            new_dict[key[13:]] = state_dict[key]
    cls.load_state_dict(new_dict)
    return cls


def load_or_learn_mapping(mapping_network=None, gaussian_fit_loc=None, 
                          device='cuda', num_samples=100000, z_dim=512, relu_alpha=5):
    try:
        return MappingProxy(torch.load(gaussian_fit_loc, map_location='cpu'))
    except:
        mapping_network.to(device)
        latent = torch.randn((num_samples, z_dim), dtype=torch.float32, device=device)
        out = torch.nn.LeakyReLU(relu_alpha)(mapping_network(latent, None))
        gaussian_fit = {"mean": out.mean((0, 1)), "std": out.std((0, 1))}
        torch.save(gaussian_fit, gaussian_fit_loc)
        return MappingProxy(torch.load(gaussian_fit_loc, map_location='cpu'))


def create_folder(folder):
    if os.path.isdir(folder):
        while 1:
            response = input('Directory exists. Do you want to overwrite? (y/N) ')
            if response == 'y':
                shutil.rmtree(folder)
                break
            elif response == 'N':
                os._exit(0)
    os.makedirs(folder)


def save_images(samples, loc, normalize=False):
    torchvision.utils.save_image(
        samples,
        loc,
        nrow=int(samples.shape[0] ** 0.5),
        normalize=normalize,
        scale_each=True)

def load_dict(model, ckpt, device='cuda'):
    state_dict = torch.load(ckpt, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        print('Loading model failed... Trying to remove the module from the keys...')
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_state_dict[key[len('module.'):]] = value
        model.load_state_dict(new_state_dict)
    return model


def to_rgb(img, old_min=-1, old_max=1):
    return (255 * (img - old_min) / (old_max - old_min + 1e-5)).to(torch.uint8)

def get_time():
    return datetime.now().strftime('%H:%M:%S:%f')

def get_prob_from_bpd(bpd):
    return math.exp(-math.log(bpd))


def make_noise(bs, device='cuda'):
    noises = [torch.randn(bs, 1, 2 ** 2, 2 ** 2, device=device)]
    for i in range(3, 11):
        for _ in range(2):
            noises.append(torch.randn(bs, 1, 2 ** i, 2 ** i, device=device))

    return noises


def mean_latent(generator, n_latent, device='cuda'):
    try:
        style_dim = generator.style_dim
    except:
        style_dim = generator.module.style_dim
    
    latent_in = torch.randn(
        n_latent, style_dim, device=device
    )
    try:
        latent = generator.style(latent_in).mean(0, keepdim=True)
    except:
        latent = generator.module.style(latent_in).mean(0, keepdim=True)
    return latent

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    new_list = []
    for i in range(0, len(lst), n):
        new_list.append(lst[i:i + n])
    return new_list

def load_stylegan_images(image_res, project_dir, input_files, device='cuda'):
    transform = get_transformation(image_res)
    imgs = []
    for img_index, imgfile in enumerate(input_files):
        imgs.append(transform(Image.open(os.path.join(project_dir, imgfile)).convert("RGB")))
    imgs = torch.stack(imgs, 0).to(device)
    return imgs


def get_mask(config, img, zero_val=0):
    # mask the totally black pixels
    if config['mask_black_pixels'] and sum(config['mse']) + sum(config['pe']) > 0:
        curr_shape = img.shape
        mask = torch.ones(curr_shape, device=config['device'])
        bs, x, y = torch.where(img.sum(dim=1) <= (zero_val + 1e-1))
        mask[bs, :, x, y] = 0
    else:
        mask = None
    return mask

def get_eval_loss_fn(config):
    device = config['device']
    functions = []
    labels = []
    if 'mse' in config:
        labels.append('ref_MSE')
        functions.append(lambda x, y: F.mse_loss(x, y))
    
    if 'pe' in config:
        labels.append('ref_LPIPS')
        # image to 256
        downsampler = BicubicDownSample(config['image_size'][0] // 256, 
                                        device=config['device'])
        percept_fn = lpips.PerceptualLoss(model="net-lin", net="vgg", gpu_ids=[device])
        functions.append(lambda x, y: percept_fn(downsampler(x), downsampler(y)).mean())

    return (lambda x, y: [fn(x, y) for fn in functions]), labels




def get_loss_fn(config, index, latent_z=None, mask=None):
            
    device = config['device']
    functions = []
    labels = []
    if 'mse' in config and config['mse'][index] > 0:
        labels.append('MSE')
        if mask is not None:
            def mask_mse_loss(x, y):
                return config['mse'][index] * F.mse_loss(x * mask.unsqueeze(1), y * mask.unsqueeze(1))
            functions.append(mask_mse_loss)    
        else:
            functions.append(lambda x, y: config['mse'][index] * F.mse_loss(x, y))
    
    if 'pe' in config and config['pe'][index] > 0:
        labels.append('LPIPS')
        # image to 256
        downsampler = BicubicDownSample(config['image_size'][0] // 256, 
                                        device=config['device'])
        percept_fn = lpips.PerceptualLoss(model="net-lin", net="vgg", gpu_ids=[device])
        if mask is None:
            functions.append(lambda x, y: config['pe'][index] * percept_fn(downsampler(x), downsampler(y)).mean())
        else:
            functions.append(lambda x, y: config['pe'][index] * percept_fn(downsampler(x), downsampler((1 -  mask) * x + mask * y)).mean())
    if 'clip' in config and config['clip'][index] > 0:
        labels.append('CLIP')
        clip_model, _ = clip.load("ViT-B/32", jit=False)
        clip_model.to(config['device'])
        clip_upsampler = torch.nn.Upsample(scale_factor=7).to(device)
        clip_pooler = torch.nn.AvgPool2d(kernel_size=32).to(device)
        text = torch.cat([clip.tokenize(config['clip_texts'])]).to(config['device'])
        def clip_fn(x, y):
            clip_loss = clip_model(clip_pooler(clip_upsampler(x)), text)
            clip_loss = 1 - clip_loss[0].mean()
            return config['clip'][index] * clip_loss
        functions.append(clip_fn)

    if 'cls' in config and config['cls'][index] > 0:
        labels.append('ImageNet Classifier')
        cls = imagenet_models.resnet50()
        cls = load_classifier_from_robustness(cls, config['cls_path']).to(device)
        # image to 256
        downsampler = BicubicDownSample(config['image_size'][0] // 256, 
                                        device=config['device'])
        functions.append(lambda x, y: F.cross_entropy_loss(config['cls'][index] * cls(downsampler(x)), 
                                                           config['target'] * torch.ones(cls.shape[0], 
                                                                                         device=x.device, 
                                                                                         dtype=torch.int64)))
    if 'geocross' in config and latent_z is not None and config['geocross'] > 0:
        labels.append('GEO')
        functions.append(lambda x, y: loss_geocross(latent_z[:, 2 * index:]) * config['geocross'])
    return (lambda x, y: [fn(x, y) for fn in functions]), labels

def get_sde(config):
    if config['sde_type'] == 'VE_SDE':
        sde = VESDE(sigma_min=config['VE_SDE']['sigma_min'],
                     sigma_max=config['VE_SDE']['sigma_max'],
                     N=config['VE_SDE']['num_scales'])
    elif config['sde_type'] == 'VP_SDE':
        sde = VPSDE(beta_min=config['VP_SDE']['beta_min'],
                    beta_max=config['VP_SDE']['beta_max'],
                    N=config['VP_SDE']['num_scales'])
    elif config['sde_type'] == 'SubVP_SDE':
        sde = SubVPSDE(beta_min=config['SubVP_SDE']['beta_min'],
                       beta_max=config['SubVP_SDE']['beta_max'],
                       N=config['SubVP_SDE']['num_scales'])
    else:
        raise Exception(f'SDE {config["sde_type"]} not supported...')
    return sde

def initialize_score_model(config, ckpt=None):    
    sde = get_sde(config)
    score_model = ViT(
        height=config['image_shape'][-2],
        width=config['image_shape'][-1],
        channels=512,
        patch_size=config['patch_size'],
        dim=config['d_model'],
        depth=config['N'],
        heads=config['heads'],
        mlp_dim=config['d_model'] * 4,
        dropout=0.0,
        emb_dropout=0.0).to(config['device'])
    if ckpt is not None:
        state_dict = torch.load(ckpt, map_location='cpu')
        load_dict(score_model, state_dict, device=config['device'])
    return score_model, sde


def get_sampler(config):
    samplers = {
    'ode': functools.partial(ode_sampler,
                            rtol=config['error_tolerance'],
                            atol=config['error_tolerance'],
                            eps=config['eps']),
    'euler': functools.partial(Euler_Maruyama_sampler,
                               num_steps=config['num_steps'],
                               eps=config['eps']),
    'distribution': functools.partial(euler_distribution_sampler,
                              num_steps=config['num_steps'],
                              step_size=config['step_size'],
                              eps=config['eps']),
    'pc': functools.partial(pc_sampler, 
                            snr=config['signal_to_noise_ratio'],
                            num_steps=config['num_steps'],
                            eps=config['eps'],
                            verbose=True),
    }
    sampler = samplers[config['sampler']]
    return sampler




def mp_setup(rank, world_size, port=None):
    if port is None:
        port = '12345'
    else:
        port = str(port)
    if os.getenv("SLURM_NODEID") is None:
        global_rank = 0
    else:
        global_rank = int(os.environ['SLURM_NODEID']) * torch.cuda.device_count() + rank
    if os.getenv("MASTER_ADDR") is None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size, init_method='env://')
    torch.cuda.set_device(rank)
    return global_rank


def mp_cleanup():
    dist.destroy_process_group()


def update_pbar_desc(pbar, metrics, labels):
    pbar_string = ''
    for metric, label in zip(metrics, labels):
        if isinstance(metric, torch.Tensor):
            pbar_string += f'{label}: {abs(metric.item()):.7f}; '
        else:
            pbar_string += f'{label}: {abs(metric):.7f}; '
    pbar.set_description(pbar_string)

def normalize(x, min, max):
    return (x - min) / (max - min + 1e-8)

def denormalize(y, min, max):
    return y * (max - min + 1e-8) + min



# normalization constants, computed from the dataloader

# N8H1 run
# gen_min = -2.7165
# gen_max = 14.9458
# latent_min = -5.7815
# latent_max = 5.8010

# N4H4 run
gen_min = -2.7126
gen_max = 15.3400
latent_min = -5.6777
latent_max = 5.8902

def unpack(x, config):
    img_dim = config['image_shape'][-1] - 1
    bs = x.shape[0]
    x  = x.reshape(bs, 512, -1)
    gen_out = x[:, :, :img_dim ** 2].reshape(bs, 512, img_dim, img_dim)
    latent_z = x[:, :, img_dim**2:img_dim**2 + 18].permute(0, 2, 1)
    return denormalize(gen_out, gen_min, gen_max), denormalize(latent_z, latent_min, latent_max)

def pack(gen_out, latent_z, config):
    img_dim = config['image_shape'][-1] - 1
    bs = gen_out.shape[0]
    return torch.cat([normalize(gen_out, gen_min, gen_max).reshape(bs, 512, -1), 
                     normalize(latent_z, latent_min, latent_max).permute(0, 2, 1), 
                     torch.zeros(bs, 512, (img_dim + 1) ** 2 - img_dim ** 2 - 18, device='cuda')
                     ], 2).reshape(bs, 512, img_dim + 1, img_dim + 1).to('cuda')  

class MpLogger:
    def __init__(self, logger, rank):
        self.logger = logger
        self.rank = rank

    def info(self, message):
        self.logger.info(f'{[self.rank]}, ' + message)


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.
  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.
  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.
    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)

  def score_fn(x, t):
    # Scale neural network output by standard deviation and flip sign
    if continuous or isinstance(sde, sde_lib.SubVPSDE):
      std = sde.marginal_prob(torch.zeros_like(x), t)[1]
    else:
      raise NotImplementedError('Non continuous models not yet implemented...')
    score = model_fn(x, t)
    score = -score / std[:, None, None, None]
    return score

  return score_fn

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def get_model_device(model):
    return next(model.parameters()).device

def get_start_layer_from_dim(dim):
    return int(math.log2(dim)) - 1

def read_yaml(path):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def get_forward_operator(operators_config, device, index=None):
    type = operators_config['operator_type']
    if type == 'dummy':
        return DummyForwardOperator()
    elif type == 'downsampler':
        return DownsamplerForwardOperator(operators_config['downsampler']['factor'])
    elif type == 'circulant':
        # if isinstance(operators_config['circulant']['m'], list):
        #     m = operators_config['circulant']['m'][index]
        # else:
        m = operators_config['circulant']['m']
        return CirculantForwardOperator(m, device=device)
    elif type == 'colorizer':
        return ColorizerForwardOperator()
    else:
        raise NotImplementedError(f'{type} operator is not currently implemented.')

class ForwardOperator:
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class DummyForwardOperator(ForwardOperator):
    def forward(self, x):
        return x

class DownsamplerForwardOperator(ForwardOperator):
    def __init__(self, factor):
        self.downsampler = BicubicDownSample(factor)

    def forward(self, x):
        return self.downsampler(x)


class ColorizerForwardOperator(ForwardOperator):
    def forward(self, x):
        # import pdb; pdb.set_trace()
        y = (0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]).unsqueeze(1).repeat(1, 3, 1, 1)
        return y
        
class CirculantForwardOperator(ForwardOperator):
    def __init__(self):
        super().__init__()
    def __init__(self, m, device='cuda'):
        self.indices = torch.tensor(np.random.choice(np.arange(1024 * 1024 * 3), int(m), replace=False)).to(device)
        self.filters = torch.ones((1024 *  1024 * 3)).normal_().unsqueeze(0).to(device)
        self.sign_pattern = (torch.rand(1024 * 1024 * 3) > 0.5).type(torch.int32).to(device)
        self.sign_pattern = 2 * self.sign_pattern - 1

    def forward(self, inputs):
        filters = self.filters
        indices = self.indices
        sign_pattern = self.sign_pattern

        n = np.prod(inputs.shape[1:])
        bs = inputs.shape[0]
        input_reshape = inputs.reshape(bs, n)
        input_sign = input_reshape * sign_pattern
        def to_complex(tensor):
            zeros = torch.zeros_like(tensor)
            concat = torch.cat((tensor, zeros), axis=0)
            reshape = concat.view(2, -1, n)
            return reshape.permute(1, 2, 0)
        complex_input = to_complex(input_sign)
        complex_filter = to_complex(filters)
        input_fft = torch.fft.fft(complex_input, 1)
        filter_fft = torch.fft.fft(complex_filter, 1)
        output_fft = input_fft * filter_fft
        output_ifft = torch.fft.ifft(output_fft, 1)
        output_real = torch.view_as_real(output_ifft)[:, :, 0, 0]
        return output_real[:, indices]
