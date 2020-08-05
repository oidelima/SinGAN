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
import gc
import GPUtil
# import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def train(opt,Gs,Zs,reals, crops, masks, eyes, NoiseAmp):

    real_ = functions.read_image(opt)
    real = imresize(real_,opt.scale1,opt)
    
    mask_ = functions.read_mask(opt) 
    crop_ = torch.zeros((1,1,opt.crop_size, opt.crop_size)) #Used just for size reference when downsizing
    crop_ = imresize(crop_,opt.scale1,opt)
    eye_ = functions.generate_eye_mask(opt, mask_, 0)
    eye_color = functions.get_eye_color(real)
    eye_color = [241, 238, 240]
    
    # eye_color = [255, 255, 255]
    opt.eye_color = eye_color
    
    in_s = 0
    scale_num = 0
    reals = functions.create_pyramid(real,reals, opt)
    masks = functions.create_pyramid(mask_,masks,opt, mode = "mask")
    eyes = functions.create_pyramid(eye_,eyes,opt, mode = "mask")
    
    

    # plt.imshow(masks[-1].cpu().detach().squeeze(), cmap="gray")
    # plt.show()
    # plt.imshow(masks[-2].cpu().detach().squeeze(), cmap="gray")
    # plt.show()
    # plt.imshow(masks[1].cpu().detach().squeeze(), cmap="gray")
    # plt.show()
    # masks[1] = functions.make_binary(masks[1], opt)
    # # im[im<=0.1] = 0
    # plt.imshow(masks[1].cpu().detach().squeeze(), cmap="gray")
    # plt.show()
    # ds
    
    # opt.eye_rho = 0

  
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
        # print(torch.cuda.memory_summary(device=opt.device, abbreviated=False))
        print("WORKED 1")
        D_curr,G_curr = init_models(opt)
        print("FINISHED 1")
        

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
        real, _, _ = functions.random_crop(real_fullsize, crop_size, opt)
    else:
        real = real_fullsize.clone()  
        real = real.repeat(opt.batch_size, 1, 1, 1)
        
        
    real = real.half()
    mask = masks[len(Gs)].half()
    eye = eyes[len(Gs)].half()
    #eye = functions.generate_eye_mask(opt, masks[-1], opt.stop_scale - len(Gs))
    
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer) width 
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer) height
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride 
    #3, 5, 7, 9, (11, 13, 15)
    #5, 7, 9, 11, 13, 15
    #10, 12, 14, 16, 18, 20

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
    
    
    
    # niter=1 if len(Gs) < 5 else 5000
    # for epoch in range(opt.niter):
    for epoch in range(opt.niter):
        
        if epoch % 1 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

                
        # opt.eye_rho += 1/18000
        # opt.eye_rho = 1
        
        
        if opt.resize:
            max_patch_size = int(min(real.size()[2], real.size()[3],mask.size()[2]*1.25))
            min_patch_size = int(max(mask.size()[2] * 0.75, 1))
            patch_size = random.randint(min_patch_size, max_patch_size)
            mask_in = nn.functional.interpolate(mask.clone(), size=patch_size) 
            eye_in = nn.functional.interpolate(eye.clone(), size=patch_size)
        else:
            mask_in = mask
            eye_in = eye

        
                        
        # eye_colored = eye_in.clone() 
        
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
            real_output = output
            #D_real_map = output.detach()
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # print(0)

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
                
            # print("0.1")

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev

            # print("0.2")

            # Stacking masks and noise to make input
            # G_input = functions.make_input(noise, mask_in, eye_in, opt)   
            # print("0.25")            
            fake_background = netG(noise.detach().half(),prev)
            # print("0.3")
            


            # plt.imsave('eye_mask.png', functions.convert_image_np(G_input[0:1, 4:5, :, :].detach()))
            # plt.imsave('fake_ind.png', functions.convert_image_np(G_input[0:1, 3:4, :, :].detach()))
            # plt.imsave('noise.png', functions.convert_image_np(G_input[0:1, 0:3, :, :].detach()))
            
            
            # import copy
            # netG_copy = copy.deepcopy(netG)
              
            # Cropping mask shape from generated image and putting on top of real image at random location
            #fake, fake_ind = functions.gen_fake(real, fake_background, mask_in, opt)
            fake, fake_ind, eye_ind, mask_ind = functions.gen_fake(real, fake_background, mask_in, eye_in, opt.eye_color, opt)
            fake = fake.half()
            # print("0.4")

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
            # print("0.5")
            # eye_ind = functions.make_binary(eye_ind, opt)

            # zeros = torch.zeros_like(eye_ind)
            weights = torch.ones(1, 1, opt.receptive_field, opt.receptive_field)/opt.receptive_field
            # print("0.6")

            mask_down = nn.functional.conv2d(mask_ind, weights).to(opt.device)
            const_down = nn.functional.conv2d(eye_ind[:,0:1,:,:], weights).to(opt.device)
            # print("0.7")
            


            

            # mask_down = nn.functional.interpolate(mask_ind.to(opt.device), size=(output.size()[2], output.size()[3]))
            # print(mask_down.size())
            # const_down = nn.functional.interpolate(eye_ind.to(opt.device), size=(output.size()[2], output.size()[3]))
            # print(const_down.size())
            

            # if opt.upweight:
            #     mask_down = nn.functional.interpolate(mask_ind.to(opt.device), size=(output.size()[2], output.size()[3]))
            #     const_down = nn.functional.interpolate(eye_ind.to(opt.device), size=(output.size()[2], output.size()[3]))
            #     # num_pix = output.size()[0] * output.size()[1] * output.size()[2] * output.size()[3] * 2 # x 2 to account for 'real' batch
            #     # num_fake = torch.sum(mask_down) 
            #     # # num_const_fake = torch.sum(const_down) 
            #     # # const_mult = num_fake/num_const_fake if num_const_fake != 0 else 0
            #     # num_real = num_pix - num_fake
            #     # mult = num_real / num_fake
            #     mult = 1.0
            #    # mask_mult = ((1-mask_down) + mask_down*mult)

            #     # mask_mult = ((1-mask_down) + mask_down*mult)# + const_mult*const_down)
            #     mask_mult =  ((1-const_down) + const_down*mult)
            #     # plt.imsave('mask_mult.png', mask_mult[0,-1,:,:].detach().cpu().numpy())
            #     # plt.imsave('output0.png', mask_mult[1,-1,:,:].detach().cpu().numpy())
            #     # plt.imsave('output1.png', mask_mult[2,-1,:,:].detach().cpu().numpy())
            #     # plt.imsave('output0.png', mask_mult[1,-1,:,:].detach().cpu().numpy())
            #     # plt.imsave('output1.png', mask_mult[2,-1,:,:].detach().cpu().numpy())
            #     # print(output.size())
            #     # print(mask_mult.size())
            #     # print(mask_mult[0, :, :, :].sum())
            #     # print(mask_mult[1, :, :, :].sum())
            #     output = output * mask_mult

            
            
            
              
            # errD_fake = output.mean()
            # num_fake = torch.sum(mask_down)
            # num_pix = output.size()[0] * output.size()[1] * output.size()[2] * output.size()[3] *2
            # num_real = num_pix - num_fake
            # mult = num_real / num_fake
            # if len(Gs) < 2:
            #     errD_fake = (output*mask_down).sum()/mask_down.sum() - (output*(1-mask_down)).sum()/(1-mask_down).sum()
            # print("1")
            errD_fake = (output*const_down).sum()/const_down.sum()+(output*mask_down).sum()/mask_down.sum()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()
            # print("2")

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            # print("3")
            gradient_penalty.backward()
            # print("4")
            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()
            del fake_background, noise
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # print("5")

        # errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):

            netG.zero_grad()
            output = netD(fake.half())
            # print(output.size())
            # print(fake.size())
            # print(mask_in.size())
            # print(real.size())
            print("6")
            if opt.upweight: output = output*mask_mult
            #D_fake_map = output.detach()
            eye_colored = eye_ind.clone().to(opt.device)
            eye_colored[:, 0, :, :] *= (opt.eye_color[0]/255)
            eye_colored[:, 1, :, :] *= (opt.eye_color[1]/255)
            eye_colored[:, 2, :, :] *= (opt.eye_color[2]/255)
            
            print("7")

            # print(fake*eye_ind.to(opt.device))
            # L1_eye_loss = 0.3*abs(fake*eye_ind.to(opt.device) - eye_colored) #*(1*len(Gs))
            # errG = -output.mean() 
            # errG = -(output*mask_down).sum()/mask_down.sum() + L1_eye_loss.sum()#+ (output*(1-mask_down)).mean()
            # eye_output = output*const_down
            # diff = output*(1-const_down)
            

            errG = -(output*const_down).sum()/const_down.sum() - (output*mask_down).sum()/mask_down.sum()
            print("8")
            errG.backward(retain_graph=True)
            print("9")
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
            torch.cuda.empty_cache()
            # torch.cuda.synchronize()
        
        # errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        

        # if epoch % 250 == 0 or epoch == (opt.niter-1):

        #     # fake_with_mask = (fake*(1-eye_ind.to(opt.device)) + eye_colored)


        #     plt.imsave('%s/fake_sample_%s.png' %  (opt.outf, epoch), functions.convert_image_np(fake[0:1, :, :, :].detach()))
        #     plt.imsave('%s/fake_indicator_%s.png' %  (opt.outf, epoch), functions.convert_image_np(fake_ind[0:1, :, :, :].detach()))
        #     plt.imsave('%s/eye_indicator_%s.png' %  (opt.outf, epoch), functions.convert_image_np(eye_ind[0:1, :, :, :].detach()))
        #     plt.imsave('%s/background_%s.png' %  (opt.outf, epoch ), functions.convert_image_np(fake_background[0:1, :, :, :].detach()))
        #     plt.imsave('%s/fake_discriminator_heat_map_%s.png' %  (opt.outf, epoch), output[0, -1, :, :].detach().cpu().numpy())
        #     plt.imsave('%s/real_discriminator_heat_map_%s.png' %  (opt.outf, epoch), real_output[0, -1, :, :].detach().cpu().numpy())
        #     # plt.imsave('%s/eye_output_%s.png' %  (opt.outf, epoch), eye_output[0, -1, :, :].detach().cpu().numpy())
        #     # plt.imsave('%s/diff_%s.png' %  (opt.outf, epoch), diff[0, -1, :, :].detach().cpu().numpy())
        #     # plt.imsave('%s/fake_with_eye%s.png' %  (opt.outf, epoch), functions.convert_image_np(fake_with_mask[0:1, :, :, :].detach()))
     

        #     plt.plot(errD2plot)
        #     plt.savefig('%s/errD.png' %  (opt.outf))
        #     plt.close()
        #     plt.plot(errG2plot)
        #     plt.savefig('%s/errG.png' %  (opt.outf))
        #     plt.close()
        #     plt.plot(D_real2plot)
        #     plt.savefig('%s/error_D_real.png' %  (opt.outf))
        #     plt.close()
        #     plt.plot(D_fake2plot)
        #     plt.savefig('%s/error_D_fake.png' %  (opt.outf))
        #     plt.close()
        #     plt.plot(z_opt2plot)
        #     plt.savefig('%s/rec_loss.png' %  (opt.outf))
        #     plt.close()


 
        #     plt.imsave('%s/mask_down_%s.png' %  (opt.outf, epoch), mask_down[0, -1, :, :].detach().cpu().numpy(), cmap="gray")
        #     plt.imsave('%s/const_down_%s.png' %  (opt.outf, epoch), const_down[0, -1, :, :].detach().cpu().numpy(), cmap="gray")
        #     # plt.imsave('%s/mask_mult_%s.png' %  (opt.outf, epoch), mask_mult[0, -1, :, :].detach().cpu().numpy())
        #     #plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(input_opt.detach(), z_prev).detach()))
        #     #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
        #     #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
        #     #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
        #     #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
        #     #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
        #     #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


        #     torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))
        
        schedulerD.step()
        schedulerG.step()
        
        if opt.random_crop:
            real, _, _ = functions.random_crop(real_fullsize, crop_size, opt)  #randomly find crop in image
            real = real.half()
        # if opt.random_eye:
        #     eye = functions.generate_eye_mask(opt, masks[-1], opt.stop_scale - len(Gs)).to(opt.device)
        
        
        # print("Epoch = ", epoch, " Level= ", len(Gs))
        # GPUtil.showUtilization()
        
        del fake, fake_ind, eye_ind, eye_colored, output, real_output, mask_down, mask_ind
        gc.collect()
        # torch.cuda.empty_cache()
    functions.save_networks(netG,netD,z_opt,opt)
    # print("EYE RHO: ", opt.eye_rho)
    # if len(Gs) == (opt.stop_scale):
    #     netG = netG_copy
            
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
    
    # def setup(rank, world_size):
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '12355'

    #     # initialize the process group
    #     dist.init_process_group("gloo", rank=rank, world_size=world_size)

    #generator initialization:
    
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG = netG.half()
    netG = nn.DataParallel(netG,device_ids=[0])
    # netG = DDP(netG,device_ids=[0])
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)
    for layer in netG.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD = netD.half()
    netD = nn.DataParallel(netD,device_ids=[0])
    # netG = DDP(netG,device_ids=[0])
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    # print(netD)
    for layer in netD.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    return netD, netG
