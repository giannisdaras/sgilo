from comet_ml import Experiment, ExistingExperiment
import numpy as np
import math
import torch
from torch import optim
import torch_utils
from tqdm import tqdm
import time
import torch.nn as nn
from torch import nn
import hydra
import os
import logging
import clip
import random
from collections import defaultdict
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import ImageDataset
import multiprocessing
from stylegan.networks import Generator
from torch.utils.data.distributed import DistributedSampler
from utils import *
from score.samplers import *
from score.likelihood import *
import math
from omegaconf import OmegaConf

class LatentOptimizer(torch.nn.Module):
    def __init__(self, config, logger, project_dir='./', experiment=None):
        super().__init__()
        
        self.config = config
        if config['image_size'][0] != config['image_size'][1]:
            raise Exception('Non-square images are not supported yet.')
        
        self.logger = logger
        
        device = config['device']
        self.project_dir = project_dir
        
        # load generator and mapping network
        with open(os.path.join(project_dir, config['gen_ckpt']), 'rb') as f:
            G = pickle.load(f)['G_ema']
            self.gen = Generator(G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels).synthesis
            self.gen.load_state_dict(G.synthesis.state_dict())
            self.gen = self.gen.to(device)
        
        if self.config['mapping_network']:
            def simulate_mapping(latent_z):
                G.mapping.to(device)
                w = G.mapping(latent_z[0], None)
                return w[:, 0, :].unsqueeze(0)
            self.mpl = lambda x: simulate_mapping(x)
        else:
            self.mpl = load_or_learn_mapping(gaussian_fit_loc=os.path.join(project_dir, config['gaussian_fit'])).to(device)
            del G
        self.experiment = experiment

        # there is a single latent model, which we retrieve from its config
        if sum(self.config['latent_temperature']) > 0:
            latent_config = read_yaml(
                os.path.join(self.project_dir, 'configs/trained_networks', self.config['latent_config'])
            )
            self.latent_score_model, self.latent_sde = initialize_score_model(latent_config)
            self.latent_score_model.to(device)
            load_dict(self.latent_score_model, os.path.join(project_dir, latent_config['ckpt']))
        
    # this function determines which noises are getting updated
    def _update_buf_grads(self, noise_list):
        for flag, buf in zip(noise_list, self.buffers):
            buf.requires_grad = True if flag else False
    
    # initializes latents and noises for optimization
    def _initialize(self, bs):
        _, labels = get_eval_loss_fn(self.config)
        
        self.best_eval_metrics = {}
        for label in labels:
            self.best_eval_metrics[label] = math.inf

        self.gen_outs = [None]    

        
        self.latent_z = torch.ones(
                    (bs, 18, 512),
                    dtype=torch.float,
                    requires_grad=True,
                    device=self.config['device'])

        # self.latent_z = torch.randn(
        #             (bs, 18, 512),
        #             dtype=torch.float,
        #             requires_grad=True,
        #             device=self.config['device'])
        
        if self.config['latents'] is not None:
            self.latent_z = torch.load(os.path.join(self.project_dir, self.config['latents']))

        self.buffers = [x for x in { name: buf for (name, buf) in self.gen.named_buffers() if 'noise_const' in name }.values()]
        for buf in self.buffers:
            if len(buf.shape) == 2:
                buf.data = torch.randn([bs] + list(buf.shape), device=buf.device)
            else:
                buf.data = torch.randn_like(buf, device=buf.device)

    # inversion for a specific layer
    def _invert(self, y, start_layer, noise_list, 
                steps, index, mask=None, refs=None):
        self.logger.info(f"Running round {index + 1} / {len(self.config['steps'])} of ILOv2.")
        # if we don't want a forward operator, we can specify the `dummy` operator in the config
        self.operator = get_forward_operator(self.config['operators'], 
                                             device=self.config['device'], 
                                             index=index)
        if self.config['operators']['forward_measurements']:
            out = self.operator(y)
            out = transforms.Resize(y.shape[-2])(out)
            save_images(out, f'{self.config["exp_name"]}_measurements.jpg', normalize=True)

        
        # based on the layer we are, there is a different score-network that controls the
        # intermediate output. Each network is retrieved through its config.
        if self.config['gen_temperature'][index] > 0:
            gen_dict = read_yaml(
                os.path.join(self.project_dir, 'configs/trained_networks', self.config['gen_configs'][index])
            )
            self.gen_score_model, self.gen_sde = initialize_score_model(gen_dict)
            self.gen_score_model.to(self.config['device'])
            load_dict(self.gen_score_model, os.path.join(self.project_dir, gen_dict['ckpt']))


        # set up optimizer by setting which noises we are going to optimize and 
        # setting the vars.
        self._update_buf_grads(noise_list)
        var_list = []
        with torch.no_grad():
            if 'n' in self.config['vars'][index]:
                var_list += self.buffers
            if 'l' in self.config['vars'][index]:
                var_list.append(self.latent_z)
            if self.gen_outs[-1] is not None and 'g' in self.config['vars'][index]:
                var_list.append(self.gen_outs[-1])
        
        learning_rate = self.config['lr'][index]
        optimizer = optim.Adam(var_list, lr=learning_rate)
        
        # If not empty, we will do Projected Gradient Descent
        unit_projector_var_list = []
        if self.config['project_latent']:
            unit_projector_var_list.append(self.latent_z)
        if self.config['project_noises']:
            unit_projector_var_list += self.buffers
        unit_projector = SphericalOptimizer(unit_projector_var_list)

        pbar = tqdm(range(steps), disable=(self.config['device'] != 0))
        
        # composes different losses
        self.loss_fn, self.labels = get_loss_fn(
            self.config, 
            index, 
            mask=mask, 
            latent_z=self.latent_z)
        
        self.eval_loss_fn, eval_labels = get_eval_loss_fn(self.config)
        
        # prepare score-networks
        if sum(self.config['latent_temperature']) > 0:
            latent_config = read_yaml(
                os.path.join(self.project_dir, 'configs/trained_networks', self.config['latent_config'])
            )
            latent_sampling_fn = get_sampling_fn(latent_config['sampling'],
                                        self.latent_sde, 
                                        [y.shape[0]] + list(latent_config['image_shape']), 
                                        inverse_scaler=lambda x: x,
                                        verbose=False,
                                        **latent_config['sampling'])

        if self.config['gen_temperature'][index]:
            gen_config = read_yaml(
                os.path.join(self.project_dir, 'configs/trained_networks', self.config['gen_configs'][index])
            )

            gen_sampling_fn = get_sampling_fn(gen_config['sampling'], 
                                        self.gen_sde, 
                                        [y.shape[0]] + list(gen_config['image_shape']), 
                                        inverse_scaler=lambda x: x,
                                        verbose=False,
                                        **gen_config['sampling'])

        for i in pbar:
            lr = self.config['lr'][index]
            optimizer.param_groups[0]["lr"] = lr

            latent_w = self.mpl(self.latent_z)
            img_gen = self.gen(latent_w, start_layer=start_layer, layer_in=self.gen_outs[-1], noise_mode='const')
            img_gen = img_gen.clamp(-1, 1)

            out = self.operator(img_gen)           
            out = transforms.Resize(img_gen.shape[-2])(out)

            # loss computation
            if self.config['operators']['forward_measurements']:
                y_measurements = self.operator(y)
                y_measurements = transforms.Resize(img_gen.shape[-2])(y_measurements)
            else:
                y_measurements = y
            metrics = self.loss_fn(out, y_measurements)
            if refs is not None:
                with torch.no_grad():
                    eval_metrics = self.eval_loss_fn(img_gen, refs)
                
                for metric, eval_key in zip(eval_metrics, self.best_eval_metrics.keys()):
                    if metric < self.best_eval_metrics[eval_key]:
                        self.best_eval_metrics[eval_key] = metric
            else:
                eval_metrics = []

            init = normalize(self.latent_z.permute(0, 2, 1).unsqueeze(-1), 
                self.config['latent_min'], 
                self.config['latent_max'])
            

            # sampling_fn goes from x_{t} -> x_{t+1} following the vector field + noise
            # To get the score + noise, we just do x_{t+1} - x_{t} and we add this change to
            # the change that we get from the measurements error.
            if self.config['latent_temperature'][index] > 0:
                # find gradient of Langevin step
                latent_delta = denormalize(
                    latent_sampling_fn(self.latent_score_model, init)[0].squeeze(-1).permute(0, 2, 1),
                    self.config['latent_min'],
                    self.config['latent_max']) - self.latent_z
            else:
                latent_delta = 0.0
            
            
            if self.config['gen_temperature'][index] and self.gen_outs[-1] is not None:
                init = normalize(self.gen_outs[-1], self.config['gen_min'], self.config['gen_max'])
                gen_delta = denormalize(
                    gen_sampling_fn(self.gen_score_model, init)[0],
                    self.config['gen_min'],
                    self.config['gen_max']
                ) - self.gen_outs[-1]

            loss = sum(metrics)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add score to latent
            latent_noise = torch.normal(mean=0.0, std=1.0, size=self.latent_z.data.shape,
                device=self.latent_z.device) * self.config['latent_noise_temp']
            self.latent_z.data += self.config['latent_temperature'][index] * latent_delta + latent_noise
            
            if self.config['gen_temperature'][index] and self.gen_outs[-1] is not None:
                gen_noise = torch.normal(mean=0.0, std=1.0, 
                    size=self.gen_outs[-1].data.shape, device=self.gen_outs[-1].device) * self.config['gen_noise_temp']
                self.gen_outs[-1].data += self.config['gen_temperature'][index] * gen_delta + gen_noise

            if self.config['project_noises'] or self.config['project_latent']:
                unit_projector.step()
            
            update_pbar_desc(
                pbar, 
                metrics + eval_metrics, 
                self.labels + eval_labels)
        

        with torch.no_grad():
            latent_w = self.mpl(self.latent_z)
            if self.config['save_images'] and self.config['steps'][index]:
                img_gen = self.gen(latent_w, layer_in=self.gen_outs[-1], start_layer=start_layer, noise_mode='const')
                img_gen = img_gen.clamp(-1, 1)
                filename = f'{self.config["exp_name"]}_{index}.jpg'
                save_images(img_gen, filename, normalize=True)
                if self.experiment is not None:
                    self.experiment.log_image(filename)
            else:
                img_gen = None
            intermediate_out = self.gen(latent_w, layer_in=self.gen_outs[-1], start_layer=start_layer, end_layer=start_layer)
            intermediate_out.requires_grad = True
            self.gen_outs.append(intermediate_out)
        

        if self.config['clip_texts'] is None or img_gen is None:
            return 0

        self.logger.info('Verifying attributes with CLIP')
        clip_model, _ = clip.load("ViT-B/32", jit=False)
        device = img_gen.device
        clip_model.to(device)
        clip_upsampler = torch.nn.Upsample(scale_factor=7).to(device)
        clip_pooler = torch.nn.AvgPool2d(kernel_size=32).to(device)
        text = torch.cat([clip.tokenize(self.config['clip_texts'])]).to(device)
        
        def clip_fn(x):
            with torch.no_grad():
                clip_loss = clip_model(clip_pooler(clip_upsampler(x)), text)
                logit1 = clip_loss[0][0][0]
                logit2 = clip_loss[0][0][1]
                return int(logit1 > logit2)
        
        ans1 = clip_fn(img_gen)
        ans2 = clip_fn(y)
        return int(ans1 == ans2)
                



    def invert(self, y, mask=None, refs=None):
        self._initialize(y.shape[0])
        start_time = time.monotonic()

        for i, steps in enumerate(self.config['steps']):
            noise_list = [True] * (self.config['noises_lookahead'] + 2 * i)
            attr_preserved = self._invert(y, 
                start_layer=i, 
                noise_list=noise_list, 
                steps=int(steps), 
                index=i, 
                mask=mask,
                refs=refs)        
            print('seconds: ', time.monotonic() - start_time)

        list_of_latents = []
        for index in range(self.latent_z.shape[0]):
            list_of_latents.append({
                'latent_z': self.latent_z[index],
                'gen_outs': [None] + [x[index] for x in self.gen_outs[1:]],
                'bufs': [x[index] for x in self.buffers]
            })
        output_dict = {
            'list_of_latents': list_of_latents,
            'eval_metrics': self.best_eval_metrics,
        }
        return output_dict, attr_preserved


def mp_run(rank, config, project_dir, working_dir):
    global_rank = \
        mp_setup(rank, config['world_size'], port=config['port']) if config['multiprocessing'] else rank
    if global_rank == 0 and not config['debug'] and config['comet_key'] is not None:
        if config['previous_experiment'] is None:
            experiment = Experiment(config['comet_key'], 
                project_name='score-based-ilo', 
                auto_output_logging='simple')
            experiment.log_parameters(config)
            if config['experiment_tag'] is not None:
                experiment.add_tag(config['experiment_tag'])
        else:
            experiment = ExistingExperiment(
                config['comet_key'],
                previous_experiment=config['previous_experiment'],
                auto_output_logging='simple')
    else:
        experiment = None

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)
    logger = MpLogger(logger, global_rank)
    logger.info(f'Working directory: {working_dir}')
    logger.info(f'Project dir: {project_dir}')

    if global_rank == 0:
        pretty(config)

    config['device'] = rank
    # set seed for reproducibility
    # torch.manual_seed(config['seed'])
    
    attrs_preserved = 0
    total_files = 0
    for x_step, folder in zip(config['x_steps'], config['input_folders']):
        files = []
        # config['operators']['circulant']['m'] = int(x_step)
        folder_path = os.path.join(project_dir, folder)
        files = get_all_files(folder_path, pattern="*.png")
        files += get_all_files(folder_path, pattern="*.jpg")
        # files += get_all_files(folder_path, pattern="*.pt")
        total_files += len(files)
        dataset = ImageDataset(files, image_size=config['image_size'], project_dir=project_dir, input_same_as_ref=config['input_same_as_ref'])

        sampler = DistributedSampler(dataset, rank=global_rank, shuffle=True) if config['multiprocessing'] else None
        loader = torch.utils.data.DataLoader(dataset=dataset, 
                                            batch_size=min(config['batch_size'], len(dataset)),
                                            sampler=sampler,
                                            drop_last=True)
        latent_optimizer = LatentOptimizer(config, logger, project_dir, experiment=experiment)
        latent_optimizer = DDP(latent_optimizer, device_ids=[rank]).module if config['multiprocessing'] else latent_optimizer
        latent_optimizer.to(rank)

        metrics = []
        for index, dataset_item in enumerate(tqdm(loader)):
            y = dataset_item['input']
            mask = dataset_item['mask'].to(rank) if 'mask' in dataset_item else None
            refs = dataset_item['ref'].to(rank) if 'ref' in dataset_item else None
            exp_name =  str(get_time())[:18]
            if config['save_images']:
                save_images(y, f'{exp_name}_input.jpg', normalize=True)
                if refs is not None:
                    save_images(refs, f'{exp_name}_reference.jpg', normalize=True)
            latent_optimizer.config['exp_name'] = exp_name
            
            output_dict, attr_preserved = latent_optimizer.invert(y.to(rank), mask=mask, refs=refs)
            attrs_preserved += attr_preserved
            list_of_latents = output_dict['list_of_latents']
            eval_metrics = output_dict['eval_metrics']
            
            def append_to_metrics():
                values = [x for x in eval_metrics.values()]
                flag = False
                for value in values:
                    if value == math.inf:
                        flag = True
                        break

                if not flag:            
                    metrics.append([x for x in eval_metrics.values()])
            
            append_to_metrics()
        
        metrics = torch.tensor(metrics[:-1])
        metrics_mean = metrics.mean(dim=0)
        metrics_std = metrics.std(dim=0)
        mean_output_dict = {}
        std_output_dict = {}


        try:
            for index, latents in enumerate(list_of_latents):
                if config['save_latent']:
                    torch.save(latents, f'{exp_name}_{index}_latents.pt')
        except:
            logger.info('Error in saving.....')
        if index % config['save_dataloader_every'] == 0:
            torch.save(loader, f'{index}_{exp_name}_dataloader.pth')
    
    try:
        index + 1
    except:
        index = 0
    logger.info(f'Files that preserved output: {attrs_preserved}, out of {index} files.')
    if config['multiprocessing']:
        mp_cleanup()


@hydra.main(config_name='configs/solve_inverse')
def main(config):
    config = OmegaConf.to_container(config)
    working_dir = os.getcwd()
    project_dir = hydra.utils.get_original_cwd()
    if config['multiprocessing']:
        mp.spawn(mp_run,
                args=(config, project_dir, working_dir),
                nprocs=torch.cuda.device_count(), 
                join=True)
    else:
        mp_run(0, config, project_dir, working_dir)

    
if __name__ == '__main__':
    main()
