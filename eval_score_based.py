import torch
import hydra
import logging
import os
from utils import *
from score.samplers import *
from stylegan.networks import Generator
import numpy as np
from torch.nn import DataParallel
 
@hydra.main(config_name='configs/eval_score_based')
def main(config):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    logger = logging.getLogger(__name__)
    working_dir = os.getcwd()
    print(f'Working directory: {working_dir}')
    project_dir = hydra.utils.get_original_cwd()
    print(f'Project dir: {project_dir}')
    print('Running with the following config')
    pretty(config)
    logger.info(config)
    arch = config['arch']


    generator = Generator(arch['z_dim'], arch['c_dim'], arch['w_dim'], arch['img_resolution'], arch['img_channels'])
    generator.load_state_dict(torch.load(os.path.join(project_dir, config['gen_ckpt'])))
    generator = generator.to(config['device'])

    mapping_network = generator.mapping
    gen = generator.synthesis
    
    buffers = [x for x in { name: buf for (name, buf) in gen.named_buffers() if 'noise_const' in name }.values()]
    for buf in buffers:
        if len(buf.shape) == 2:
            buf.data = torch.randn([config['sample_batch_size']] + list(buf.shape), device=buf.device)
        else:
            buf.data = torch.randn_like(buf, device=buf.device)
    
    # replace the mapping network with an approximation for inverse problems
    mpl = DataParallel(load_or_learn_mapping(
        gaussian_fit_loc=os.path.join(project_dir, config['gaussian_fit']), 
        mapping_network=mapping_network)).to(config['device'])

    bs = config['sample_batch_size']
    latent_conf = read_yaml(os.path.join(project_dir, config['latent_conf']))
    gen_conf = read_yaml(os.path.join(project_dir, config['gen_conf']))
    end_layer = get_start_layer_from_dim(gen_conf['image_shape'][-1]) - 1

    # initialize 
    latent_z_init = torch.randn(bs, 18, 512, device='cuda')
    gen_out_init = gen(mpl(latent_z_init), start_layer=0, end_layer=end_layer, noise_mode=config['noise_mode'])

    # get and save initial image
    img_gen_init = gen(mpl(latent_z_init), noise_mode=config['noise_mode'])
    save_images(img_gen_init, f'stylegan.jpg', normalize=True)

    # load score-based model for latents
    latent_score_model, latent_sde = initialize_score_model(latent_conf)
    latent_score_model = latent_score_model.eval().to(config['device'])
    load_dict(latent_score_model, os.path.join(project_dir, latent_conf['ckpt']))
    latent_sampling_config = latent_conf['sampling']
    latent_sampling_fn = get_sampling_fn(latent_sampling_config, 
                                  latent_sde, [config['sample_batch_size']] + list(latent_conf['image_shape']), 
                                  inverse_scaler=lambda x: x,
                                  **latent_sampling_config)


    # load score-based model for gen out
    gen_score_model, gen_sde = initialize_score_model(gen_conf)
    gen_score_model = gen_score_model.eval().to(config['device'])
    load_dict(gen_score_model, os.path.join(project_dir, gen_conf['ckpt']))
    gen_sampling_config = gen_conf['sampling']
    gen_sampling_fn = get_sampling_fn(gen_sampling_config, 
                                  gen_sde, [config['sample_batch_size']] + list(gen_conf['image_shape']), 
                                  inverse_scaler=lambda x: x,
                                  **gen_sampling_config)
    
    # sample
    for batch_index in range(config['total_batches']):
        print(f'Running {batch_index + 1}  / {config["total_batches"]}')
        with torch.no_grad():
            if config['different_samples']:
                latent_z_init = torch.randn(bs, 18, 512, device='cuda')
                gen_out_init = gen(mpl(latent_z_init), start_layer=0, end_layer=end_layer, noise_mode=config['noise_mode'])
                img_gen_init = gen(mpl(latent_z_init), noise_mode=config['noise_mode']) 

            # latent_init = normalize(latent_z_init.permute(0, 2, 1).unsqueeze(-1), config['latent_min'], config['latent_max'])
            
            # print('Sampling latent...')
            # sample, steps = latent_sampling_fn(latent_score_model, latent_init)
            # latent_z = denormalize(sample, config['latent_min'], config['latent_max']).squeeze(-1).permute(0, 2, 1)

        latent_z = latent_z_init

        with torch.no_grad():
            gen_init = normalize(gen_out_init, config['gen_min'], config['gen_max'])
            g_vectors = gen_init
            t_index = 1.0
            # t_index = config['gen_score_config']['sampling']['t_index'] # 0.0: clean image
            # z = torch.randn_like(gen_init)
            # mean, std = gen_sde.marginal_prob(gen_init, torch.tensor(t_index).to('cuda').unsqueeze(0))
            # g_vectors = mean + std[:, None, None, None] * z

            # flag_gen_prior_sampler = config['gen_score_config']['sampling']['prior_sampler']
            # if flag_gen_prior_sampler:
            #     print('Reversing gen...')
            #     z = torch.randn_like(gen_init)
            #     mean, std = sde_gen.marginal_prob(gen_init, t_index)
            #     g_vectors = mean + std[:, None, None, None] * z
            # else:
            #     g_vectors = gen_init
            print('Sampling gen...')
            sample, steps = gen_sampling_fn(gen_score_model, x=g_vectors, t_index=t_index)

            # sample, steps = gen_sampling_fn(gen_score_model, gen_init)
            gen_out = denormalize(sample, config['gen_min'], config['gen_max'])


        with torch.no_grad():
            # # get and save final image
            img_gen = gen(mpl(latent_z), start_layer=end_layer + 1, layer_in=gen_out, noise_mode=config['noise_mode']) 
            save_images(img_gen, f'score_{batch_index}.jpg', normalize=True)

            # get and save final image
            img_gen = gen(mpl(latent_z_init), start_layer=end_layer + 1, layer_in=gen_out, noise_mode=config['noise_mode']) 
            save_images(img_gen, f'score_latent_fixed_{batch_index}.jpg', normalize=True)

            # get and save final image
            img_gen = gen(mpl(latent_z), start_layer=end_layer + 1, layer_in=gen_out_init, noise_mode=config['noise_mode']) 
            save_images(img_gen, f'score_gen_out_fixed_{batch_index}.jpg', normalize=True)

if __name__ == '__main__':
    main()