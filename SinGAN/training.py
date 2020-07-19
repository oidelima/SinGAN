import SinGAN.functions as functions
import SinGAN.models as models
#import SinGAN.manipulate as manipulate
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from SinGAN.imresize import imresize


def train(opt,Gs,Zs,reals, crops, masks, eyes, NoiseAmp):

    real_ = functions.read_image(opt)
    real = imresize(real_,opt.scale1,opt)
    
    mask_ = functions.read_mask(opt) 
    crop_ = torch.zeros((1,1,opt.crop_size, opt.crop_size)) #Used just for size reference when downsizing
    crop_ = imresize(crop_,opt.scale1,opt)
    eye_ = functions.generate_eye_mask(opt, mask_, 0)
    eye_color = functions.get_eye_color(real)
    opt.eye_color = eye_color
    
    in_s = 0
    scale_num = 0
    reals = functions.create_pyramid(real,reals, opt)
    masks = functions.create_pyramid(mask_,masks,opt, mode = "mask")
    eyes = functions.create_pyramid(eye_,eyes,opt, mode = "mask")
    #GPUtil.showUtilization()
    

  
    # Shortcut to get sizes of corresponding crops for each scale
    crops =  functions.create_pyramid(crop_,crops, opt, mode="mask")
     
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
        D_curr,G_curr = init_models(opt)

        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

        
        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals, crops, masks, eyes, Gs,Zs,in_s,NoiseAmp,opt)
        torch.cuda.empty_cache()

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        # torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        # torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        # torch.save(reals, '%s/reals.pth' % (opt.out_))
        # torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
            
    return



def train_single_scale(netD,netG,reals, crops,  masks, eyes, Gs,Zs,in_s,NoiseAmp,opt,centers=None):
    
    
    real_fullsize = reals[len(Gs)]
    crop_size =  crops[len(Gs)].size()[2]
    fixed_crop = real_fullsize[:,:,0:crop_size,0:crop_size].repeat(opt.batch_size, 1, 1, 1)
    plt.imsave('%s/fixed_alpha_crop.png' %  (opt.outf), functions.convert_image_np(fixed_crop[0:1, :, :, :].detach()))
    
    if opt.random_crop:
        real, _, _ = functions.random_crop(real_fullsize.clone(), crop_size, opt)
    else:
        real = real_fullsize.clone()  
        real = real.repeat(opt.batch_size, 1, 1, 1)
        
    mask = masks[len(Gs)]
    eye = eyes[len(Gs)]
    #eye = functions.generate_eye_mask(opt, masks[-1], opt.stop_scale - len(Gs))
    
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer) width 
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer) height
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device, num_samp=opt.batch_size)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []
    
    
    
    
    for epoch in range(opt.niter):
        
        
        if opt.resize:
            max_patch_size = int(min(real.size()[2], real.size()[3],mask.size()[2]*1.25))
            min_patch_size = int(max(mask.size()[2] * 0.75, 1))
            patch_size = random.randint(min_patch_size, max_patch_size)
            mask_in = nn.functional.interpolate(mask.clone(), size=patch_size) 
            eye_in = nn.functional.interpolate(eye.clone(), size=patch_size)
        else:
            mask_in = mask.clone()
            eye_in = eye.clone()
                        
        eye_colored = eye_in.clone() 
        
        if opt.random_eye_color:
            opt.eye_color = functions.get_eye_color(real)
                
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device, num_samp=opt.batch_size)
            z_opt = m_noise(z_opt.expand(opt.batch_size,3,opt.nzx,opt.nzy))
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device, num_samp=opt.batch_size)
            noise_ = m_noise(noise_.expand(opt.batch_size,3,opt.nzx,opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device, num_samp=opt.batch_size)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()
            output = netD(real).to(opt.device)
            real_output = output.clone()
            #D_real_map = output.detach()
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

           

            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([opt.batch_size,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([opt.batch_size,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = functions.draw_concat(Gs,Zs,reals, crops, masks,eyes, NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    prev = m_image(prev)
                    z_prev = functions.draw_concat(Gs,Zs,reals, crops, masks,eyes, NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                    criterion = nn.MSELoss()
                    #print(z_prev.get_device())
                    #print(real.get_device())
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = functions.draw_concat(Gs,Zs,reals, crops, masks, eyes, NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                prev = m_image(prev)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev

            

            # Stacking masks and noise to make input
            G_input = functions.make_input(noise, mask_in, eye_in, opt)               
            fake_background = netG(G_input.detach(),prev)

            # plt.imsave('eye_mask.png', functions.convert_image_np(G_input[0:1, 4:5, :, :].detach()))
            # plt.imsave('fake_ind.png', functions.convert_image_np(G_input[0:1, 3:4, :, :].detach()))
            # plt.imsave('noise.png', functions.convert_image_np(G_input[0:1, 0:3, :, :].detach()))
            
            
            import copy
            netG_copy = copy.deepcopy(netG)
              
            # Cropping mask shape from generated image and putting on top of real image at random location
            #fake, fake_ind = functions.gen_fake(real, fake_background, mask_in, opt)
            fake, fake_ind, eye_ind, mask_ind = functions.gen_fake(real, fake_background, mask_in, eye_in, opt.eye_color, opt)

            # plt.imshow(fake[1, :, :, :].cpu().permute(1,2,0).detach().squeeze(), cmap="gray")
            # plt.show()
            # plt.imshow(fake[2, :, :, :].cpu().permute(1,2,0).detach().squeeze(), cmap="gray")
            # plt.show()
            # plt.imshow(fake[3, :, :, :].cpu().permute(1,2,0).detach().squeeze(), cmap="gray")
            # plt.show()
            # plt.imshow(fake[4, :, :, :].cpu().permute(1,2,0).detach().squeeze(), cmap="gray")
            # plt.show()
            # plt.imshow(fake[0, :, :, :].cpu().permute(1,2,0).detach().squeeze(), cmap="gray")
            # plt.show() 
            # plt.imsave('test.png', functions.convert_image_np(fake[0:1, :, :, :].detach()))
            # plt.imsave('fake_ind.png', functions.convert_image_np(fake_ind[0:1, :, :, :].detach()))
            # plt.imsave('eye_ind.png', functions.convert_image_np(eye_ind[0:1, :, :, :].detach()))
            
                                                                                                                                                                   
            output = netD(fake.detach())

            if opt.upweight:
                mask_down = nn.functional.interpolate(mask_ind.to(opt.device), size=(output.size()[2], output.size()[3]))
                num_pix = output.size()[0] * output.size()[1] * output.size()[2] * output.size()[3] * 2 # x 2 to account for 'real' batch
                num_fake = torch.sum(1 - mask_down) 
                num_real = num_pix - num_fake
                mult = num_real / num_fake
                mult = 1.00
                mask_mult = ((1-mask_down) + mask_down*mult) 
                # plt.imsave('mask_mult.png', mask_mult[0,-1,:,:].detach().cpu().numpy())
                # plt.imsave('output0.png', mask_mult[1,-1,:,:].detach().cpu().numpy())
                # plt.imsave('output1.png', mask_mult[2,-1,:,:].detach().cpu().numpy())
                # plt.imsave('output0.png', mask_mult[1,-1,:,:].detach().cpu().numpy())
                # plt.imsave('output1.png', mask_mult[2,-1,:,:].detach().cpu().numpy())
                # print(output.size())
                # print(mask_mult.size())
                # print(mask_mult[0, :, :, :].sum())
                # print(mask_mult[1, :, :, :].sum())
                output = output * mask_mult
                
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):

            netG.zero_grad()
            output = netD(fake)
            #D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt+z_prev
                input_opt = functions.make_input(Z_opt, mask_in, eye_in, opt)
                #rec_loss = alpha*loss(netG(input_opt.detach(),z_prev),real)
                mask_height, mask_width = mask_in.size()[2], mask_in.size()[3]
                # print(mask_in)
                # plt.imshow((netG(input_opt.detach(),z_prev)[:, :, :mask_height, :mask_width]*mask_in).cpu().detach().squeeze(), cmap="gray")
                # plt.show()
                # plt.imshow((fixed_crop[:, :, :mask_height, :mask_width]*mask_in).cpu().detach().squeeze(), cmap="gray")
                # plt.show()
                #rec_loss = alpha*loss(netG(input_opt.detach(),z_prev)[:, :, :mask_height, :mask_width]*mask_in,real[:, :, :mask_height, :mask_width]*mask_in)
                rec_loss = alpha*loss(netG(input_opt.detach(),z_prev)[:, :, :mask_height, :mask_width]*mask_in,fixed_crop[:, :, :mask_height, :mask_width]*mask_in)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()
        
        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake[0:1, :, :, :].detach()))
            plt.imsave('%s/fake_indicator.png' %  (opt.outf), functions.convert_image_np(fake_ind[0:1, :, :, :].detach()))
            plt.imsave('%s/eye_indicator.png' %  (opt.outf), functions.convert_image_np(eye_ind.detach()))
            plt.imsave('%s/background.png' %  (opt.outf), functions.convert_image_np(fake_background[0:1, :, :, :].detach()))
            plt.imsave('%s/fake_discriminator_heat_map_%s.png' %  (opt.outf, epoch), output[0, -1, :, :].detach().cpu().numpy())
            plt.imsave('%s/real_discriminator_heat_map_%s.png' %  (opt.outf, epoch), real_output[0, -1, :, :].detach().cpu().numpy())
            plt.imsave('%s/mask_down_%s.png' %  (opt.outf, epoch), mask_down[0, -1, :, :].detach().cpu().numpy())
            plt.imsave('%s/mask_mult_%s.png' %  (opt.outf, epoch), mask_mult[0, -1, :, :].detach().cpu().numpy())
            #plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(input_opt.detach(), z_prev).detach()))
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))
        
        schedulerD.step()
        schedulerG.step()
        
        if opt.random_crop:
            real, _, _ = functions.random_crop(real_fullsize, crop_size, opt)  #randomly find crop in image
        if opt.random_eye:
            eye = functions.generate_eye_mask(opt, masks[-1], opt.stop_scale - len(Gs)).to(opt.device)
        
        # del real, fake_background,fake
        # torch.cuda.empty_cache()
    functions.save_networks(netG,netD,z_opt,opt)
    
    if len(Gs) == (opt.stop_scale):
        netG = netG_copy
            
    return z_opt,in_s,netG 


def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt):

    #generator initialization:
    
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG = nn.DataParallel(netG,device_ids=[6,7,8,9])
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD = nn.DataParallel(netD,device_ids=[6,7,8,9])
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
