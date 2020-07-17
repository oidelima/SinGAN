from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import SinGAN.functions as functions
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments

def generate_gif(Gs,Zs,reals,NoiseAmp,opt,alpha=0.1,beta=0.9,start_scale=2,fps=10):

    in_s = torch.full(Zs[0].shape, 0, device=opt.device)
    images_cur = []
    count = 0

    for G,Z_opt,noise_amp,real in zip(Gs,Zs,NoiseAmp,reals):
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        nzx = Z_opt.shape[2]
        nzy = Z_opt.shape[3]
        #pad_noise = 0
        #m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
        images_prev = images_cur
        images_cur = []
        if count == 0:
            z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
            z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
            z_prev1 = 0.95*Z_opt +0.05*z_rand
            z_prev2 = Z_opt
        else:
            z_prev1 = 0.95*Z_opt +0.05*functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
            z_prev2 = Z_opt

        for i in range(0,100,1):
            if count == 0:
                z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*z_rand
            else:
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*(functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device))

            z_curr = alpha*Z_opt+(1-alpha)*(z_prev1+diff_curr)
            z_prev2 = z_prev1
            z_prev1 = z_curr

            if images_prev == []:
                I_prev = in_s
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                #I_prev = functions.upsampling(I_prev,reals[count].shape[2],reals[count].shape[3])
                I_prev = m_image(I_prev)
            if count < start_scale:
                z_curr = Z_opt

            z_in = noise_amp*z_curr+I_prev
            I_curr = G(z_in.detach(),I_prev)

            if (count == len(Gs)-1):
                I_curr = functions.denorm(I_curr).detach()
                I_curr = I_curr[0,:,:,:].cpu().numpy()
                I_curr = I_curr.transpose(1, 2, 0)*255
                I_curr = I_curr.astype(np.uint8)

            images_cur.append(I_curr)
        count += 1
    dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs('%s/start_scale=%d' % (dir2save,start_scale) )
    except OSError:
        pass
    imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (dir2save,start_scale,alpha,beta),images_cur,fps=fps)
    del images_cur

def random_crop_generate(real, mask, constraint, mask_source, crop, opt, num_samples = 20, mask_locs = None):
    #eye = functions.generate_eye_mask(opt, mask, 0).to(opt.device) #generate eye in random location
    real_fullsize = real.clone()

    for i in range(num_samples):
        crop_size = crop.size()[2]
        
        if opt.random_crop:
            fake_background, _, _ = functions.random_crop(real_fullsize, crop_size, opt)
            real, h_idx, w_idx = functions.random_crop(real_fullsize.clone(), crop_size, opt)
        else:
            real = real_fullsize.clone()
            fake_background = real_fullsize.clone()

        # if opt.random_eye_color:
        #     opt.eye_color = functions.get_eye_color(real)

        #mask_loc = mask_locs[i] if mask_locs else None
        I_curr, fake_ind, constraint_ind, _ = functions.gen_fake(real, fake_background, mask, constraint, mask_source, opt, border = True, mask_loc = None)
        if opt.random_crop:
            full_fake = real_fullsize.clone()
            full_fake[:, :, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size] = I_curr[0:1, :, :, :]
            full_mask = torch.zeros_like(full_fake)
            full_mask[:, :, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size] = fake_ind[0:1, : ,:, :]
        
        dir2save = '%s/RandomSamples/%s/random_crop/%s' % (opt.out, opt.input_name[:-4], opt.run_name)
        try:
            os.makedirs(dir2save + "/fake")
            os.makedirs(dir2save + "/background")
            os.makedirs(dir2save + "/mask")
            os.makedirs(dir2save + "/constraint")
            if opt.random_crop:
                os.makedirs(dir2save + "/full_fake")
                os.makedirs(dir2save + "/full_mask") 
        except OSError:
            pass
        if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
            plt.imsave('%s/%s/%d.png' % (dir2save, "fake", i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
            plt.imsave('%s/%s/%d.png' % (dir2save, "background", i), functions.convert_image_np(real.detach()), vmin=0,vmax=1)
            plt.imsave('%s/%s/%d.png' % (dir2save, "mask", i), functions.convert_image_np(fake_ind.detach()), vmin=0,vmax=1)
            plt.imsave('%s/%s/%d.png' % (dir2save, "constraint", i), functions.convert_image_np(constraint_ind.detach()), vmin=0,vmax=1)
            if opt.random_crop:
                plt.imsave('%s/%s/%d.png' % (dir2save, "full_fake", i), functions.convert_image_np(full_fake.detach()), vmin=0,vmax=1)
                plt.imsave('%s/%s/%d.png' % (dir2save, "full_mask", i), functions.convert_image_np(full_mask.detach()), vmin=0,vmax=1)
            #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)


def SinGAN_generate(Gs,Zs,reals, crops, masks, constraints, mask_sources, NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=20, mask_locs=None):
    #if torch.is_tensor(in_s) == False:
    Gs[-1].train()
    
    if in_s == None:
        in_s  = torch.full([opt.batch_size,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
    
    if not opt.random_crop:
        real = reals[-1].repeat(opt.batch_size, 1, 1, 1).clone()
    else:
        real = reals[-1].clone()
        
    constraint = constraints[-1].clone()
    mask_source = mask_sources[-1].clone()

    for i in range(0,num_samples,1):
        
        #eye = functions.generate_eye_mask(opt, masks[-1], 0).to(opt.device) #generate eye in random location

        # if opt.random_eye_color:
        #     opt.eye_color = functions.get_eye_color(reals[-1])
        
        # eye_colored = eye.clone() 
        # if opt.random_eye_color:
        #     eye_color = functions.get_eye_color(reals[-1])
        #     opt.eye_color = eye_color
        #     eye_colored[:, 0, :, :] *= (eye_color[0]/255)
        #     eye_colored[:, 1, :, :] *= (eye_color[1]/255)
        #     eye_colored[:, 2, :, :] *= (eye_color[2]/255)
        
        noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device, num_samp=opt.batch_size)
        
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        
        m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
        
        noise_ = m_noise(noise_)
        
        
        prev = functions.draw_concat(Gs,Zs,reals, crops, masks, constraints, NoiseAmp,in_s,'rand',m_noise,m_image,opt)
        prev = m_image(prev)
        
        noise = opt.noise_amp*noise_+prev
        
        G_input = functions.make_input(noise, masks[-1], constraint, opt)
        fake_background = Gs[-1](G_input.detach(),prev)
        
        border = False #TODO
        
        if opt.random_crop:
            crop_size =  crops[-1].size()[2]
            crop, h_idx, w_idx = functions.random_crop(real, crop_size, opt)

            
            #I_curr, fake_ind = functions.gen_fake(crop, fake_background, masks[-1], opt, border, mask_loc = mask_locs[i])
            I_curr, fake_ind, constraint_ind, _ = functions.gen_fake(crop, fake_background, masks[-1], constraint, mask_source, opt, border, mask_loc = None)
            full_fake = reals[-1].clone()
            full_fake[:, :, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size] = I_curr[:1,:,:,:]
            full_mask = torch.zeros_like(full_fake)
            full_mask[:, :, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size] = fake_ind[:1, :, :, :]
        else:
            I_curr, fake_ind, constraint_ind, _ = functions.gen_fake(real, fake_background, masks[-1], constraint, mask_source, opt, border,  mask_loc = None)
            #I_curr, fake_ind = functions.gen_fake(reals[-1], fake_background, masks[-1], opt, border,  mask_loc = mask_locs[i])
        
        if opt.mode == 'train':
            dir2save = '%s/RandomSamples/%s/SinGAN/%s' % (opt.out, opt.input_name[:-4], opt.run_name)
        else:
            dir2save = functions.generate_dir2save(opt)
        try:
            os.makedirs(dir2save + "/fake")
            os.makedirs(dir2save + "/background")
            os.makedirs(dir2save + "/mask")
            os.makedirs(dir2save + "/constraint")
            if opt.random_crop:
                os.makedirs(dir2save + "/full_fake")
                os.makedirs(dir2save + "/full_mask")
            
        except OSError:
            pass
        if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
            plt.imsave('%s/%s/%d.png' % (dir2save, "fake", i), functions.convert_image_np(I_curr[:1, :, :, :].detach()))
            plt.imsave('%s/%s/%d.png' % (dir2save, "background", i), functions.convert_image_np(fake_background[:1, :, :, :].detach()))
            plt.imsave('%s/%s/%d.png' % (dir2save, "mask", i), functions.convert_image_np(fake_ind[:1, :, :, :].detach()))
            plt.imsave('%s/%s/%d.png' % (dir2save, "constraint", i), functions.convert_image_np(constraint_ind.detach()))
            if opt.random_crop:
                plt.imsave('%s/%s/%d.png' % (dir2save, "full_fake", i), functions.convert_image_np(full_fake.detach()))
                plt.imsave('%s/%s/%d.png' % (dir2save, "full_mask", i), functions.convert_image_np(full_mask.detach()))
            #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
        



        
# if in_s is None:
#     if opt.random_crop:
#         in_s = torch.full(crops[0].shape, 0, device=opt.device)
#     else:
#         in_s = torch.full(reals[0].shape, 0, device=opt.device)
# images_cur = []
# count = 0
# for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
#     eye = functions.generate_eye_mask(opt, masks[-1], opt.stop_scale - count)
#     count+=1
#     pad1 = ((opt.ker_size-1)*opt.num_layer)/2
#     m = nn.ZeroPad2d(int(pad1))
#     nzx = (Z_opt.shape[2]-pad1*2)*scale_v
#     nzy = (Z_opt.shape[3]-pad1*2)*scale_h

#     images_prev = images_cur
#     images_cur = []

#     for i in range(0,num_samples,1):
#         if n == 0:
#             z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
#             z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
#             z_curr = m(z_curr)
#         else:
#             z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
#             z_curr = m(z_curr)

#         if images_prev == []:
#             I_prev = m(in_s)
#             #I_prev = m(I_prev)
#             #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
#             #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
#         else:
#             I_prev = images_prev[i]
#             I_prev = imresize(I_prev,1/opt.scale_factor, opt)
#             if opt.mode != "SR":
#                 if opt.random_crop:
#                     I_prev = I_prev[:, :, 0:round(scale_v * crops[n].shape[2]), 0:round(scale_h * crops[n].shape[3])]
#                 else:
#                     I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
#                 I_prev = m(I_prev)
#                 I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
#                 I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
#             else:
#                 I_prev = m(I_prev)

#         if n < gen_start_scale:
#             z_curr = Z_opt

#         z_in = noise_amp*(z_curr)+I_prev.to(opt.device)
        
#         if n == len(reals)-1: 
#             border = True
            
#             z_in = torch.from_numpy(np.load("good_noise.npy"))
#             z_in = z_in.cuda()
        
#         G_input = SinGAN.functions.make_input(z_in, masks[n], eye)
        
#         fake_background = G(G_input.detach(),I_prev.to(opt.device)) 
#         plt.imsave('fake_train_2.png' , functions.convert_image_np(fake_background.detach()))
#         plt.imsave('noise_2.png' , functions.convert_image_np(z_in.detach()))
#         plt.imsave('mask_in_2.png' , functions.convert_image_np(masks[n].detach()))
#         plt.imsave('eye_in_2.png' , functions.convert_image_np(eye.detach()))
        
#         # plt.imsave('G_input_f_2.png' , functions.convert_image_np(G_input[:,:3, :, :].detach()))
#         # plt.imsave('G_input_s_2.png' , functions.convert_image_np(G_input[:,3:, :, :].detach()))
#         # np.save("fake_train_2", fake_background.cpu().detach())
#         # np.save("noise_2", z_in.cpu().detach())
#         # np.save("mask_in_2", masks[n].cpu().detach())
#         # np.save("eye_in_2", eye.cpu().detach())
        
#         border = False
        

        
        


#         mask_loc = mask_locs[i] if mask_locs and n == len(reals)-1 else None

#         if opt.random_crop:
#             crop_size =  crops[n].size()[2]
#             crop, h_idx, w_idx = functions.random_crop(reals[n], crop_size)

#             I_curr, fake_ind, eye_ind = functions.gen_fake(crop, fake_background, masks[n], eye, opt.eye_color, opt, border, mask_loc)
#             full_fake = reals[n].clone()
#             full_fake[:, :, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size] = I_curr
#             full_mask = torch.zeros_like(full_fake)
#             full_mask[:, :, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size] = fake_ind
#         else:
#             I_curr, fake_ind, eye_ind = functions.gen_fake(reals[n], fake_background, masks[n], eye, opt.eye_color, opt, border, mask_loc)

