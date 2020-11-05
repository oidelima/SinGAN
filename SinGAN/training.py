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
import math
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import cv2
import scipy



def train(opt,Gs,Zs,reals, masks, constraints, crop_sizes, mask_sources, NoiseAmp):
    
    
    # Loading input background image, normalizing and resizing to largest scale size
    real = functions.read_image(opt)
    real = imresize(real,opt.scale1,opt)
    # print(real.shape)
    # from PIL import Image, ImageDraw
    # eye = Image.new('RGB', (real.shape[3], real.shape[2]))
    # draw = ImageDraw.Draw(eye)
    # eye_loc = (real.shape[2]//2 - 1, real.shape[3]//2 - 1)
    # draw.ellipse([(eye_loc[1], eye_loc[0]), (eye_loc[1] + 3, eye_loc[0] + 3)], fill="white")
    # eye = np.array(eye)/255.0
    # print(eye.shape)
    # plt.imsave("center_reference.png" , eye)
    # functions.show_image(eye)
    
    # Loading masks and resizing so that biggest dimension is opt.patch_size 
    mask = functions.read_mask(opt,"Input/body_masks",opt.mask_name) 
    # constraint = functions.read_mask(opt, "Input/constraints", opt.mask_name) 
    
     # Loading image source for mask and resizing so that biggest dimension is opt.patch_size 
    new_dim = (mask.size()[3], mask.size()[2])
    mask_source = functions.read_image(opt, "Input/mask_sources", opt.mask_name[:-3]+"jpg", size=new_dim)
    
   
    
    # Making sure that the constraint doesn't lie outside of the body mask
    # constraint = constraint * mask #* mask_source
   
    # #test eye
    # mask_source = torch.ones_like(mask_source)

    # #ocean
    # opt.eye_diam=4
    # opt.eye_loc = (38, 78) #TODO ocean
    # mask_source[:,0,:,:]  = (241/255 - 0.5)*2
    # mask_source[:,1,:,:]  = (238/255 - 0.5)*2
    # mask_source[:,2,:,:]  = (240/255 - 0.5)*2
    
    constraint_ = functions.generate_eye_mask(opt, mask, 0, opt.fixed_eye_loc)
    constraint = constraint_ * mask #* mask_source

    in_s = 0
    scale_num = 0
    reals = functions.create_pyramid(real,reals, opt)
    masks = functions.create_pyramid(mask,masks,opt, mode = "mask")
    constraints = functions.create_pyramid(constraint, constraints,opt, mode = "mask")
    mask_sources = functions.create_pyramid(mask_source, mask_sources,opt)
    

    nfc_prev = 0
    
    opt.mask_alpha = 0


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

        
        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals,masks,constraints, mask_sources, crop_sizes, Gs,Zs,in_s,NoiseAmp,opt)


        G_curr = functions.reset_grads(G_curr,False)
        # G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        # D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        # torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        # torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        # torch.save(reals, '%s/reals.pth' % (opt.out_))
        # torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))
        # torch.save(masks, '%s/masks.pth' % (opt.out_))
        # torch.save(constraints, '%s/constraints.pth' % (opt.out_))
        # torch.save(mask_sources, '%s/mask_sources.pth' % (opt.out_))
        # torch.save(crop_sizes, '%s/crop_sizes.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
            
    return



def train_single_scale(netD,netG,reals,masks, constraints, mask_sources, crop_sizes, Gs,Zs,in_s,NoiseAmp,opt,centers=None):
    
    
    real = reals[len(Gs)]
    mask = masks[len(Gs)]
    constraint = constraints[len(Gs)]
    mask_source = mask_sources[len(Gs)]
    
    im_height, im_width = real.size()[2], real.size()[3] 
    mask_height, mask_width = mask.size()[2], mask.size()[3]
    height_init = (im_height - mask_height)//2
    width_init = (im_width - mask_width)//2
    
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
        
                   
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)
        

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):

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
                    prev = functions.draw_concat(Gs,Zs,reals,masks, constraints, mask_sources,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    prev = m_image(prev)
                    z_prev = functions.draw_concat(Gs,Zs,reals,masks, constraints, mask_sources,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init *RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = functions.draw_concat(Gs,Zs,reals,masks, constraints, mask_sources,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                prev = m_image(prev)
                
            

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev

            # G_input = functions.make_input(noise, mask_in, eye_in, opt)  
                 
            fake_background = netG(noise.detach(),prev)
            
            if opt.mode == "inpainting":
                fake, fake_ind, constraint_ind, mask_ind, constraint_filled = functions.gen_fake(real, fake_background, mask, constraint, mask_source, opt, (height_init, width_init))
            else:
                fake, fake_ind, constraint_ind, mask_ind, constraint_filled = functions.gen_fake(real, fake_background, mask, constraint, mask_source, opt)
            
            
            mask_ind = torch.round(mask_ind)

            weights = torch.ones(1, 1, opt.receptive_field, opt.receptive_field)/(opt.receptive_field**2)
            mask_down = nn.functional.conv2d(mask_ind.detach(), weights).to(opt.device)

             # train with real
            netD.zero_grad()
            output = netD(real).to(opt.device)
            real_output = output.detach()
            
            errD_real = -(output*mask_down).mean()

            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()


            output = netD(fake.detach())

            errD_fake = (output*mask_down).mean()
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

            L1_constraint_loss = 10*abs((fake_background[:,:,height_init:height_init+mask_height ,width_init:width_init + mask_width]-mask_source)*constraint.to(opt.device)).mean()
            # errG = -(output*mask_down  - (output*(1-mask_down))).mean()
            errG = -(output*mask_down).mean() + L1_constraint_loss

            errG.backward(retain_graph=True)

            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt+z_prev
                # input_opt = functions.make_input(Z_opt, mask_in, eye_in, opt)
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev),real)
                
                #rec_loss = alpha*loss(netG(input_opt.detach(),z_prev)[:, :, :mask_height, :mask_width]*mask_in,real[:, :, :mask_height, :mask_width]*mask_in)
                # rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev)[:, :, :mask_height, :mask_width]*mask,fixed_crop[:, :, :mask_height, :mask_width]*mask)
                
                # rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev)[:,:,height_init:height_init+mask_height ,width_init:width_init + mask_width]*mask,
                                    # fixed_crop[:,:,height_init:height_init+mask_height ,width_init:width_init + mask_width]*mask)

                
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

        if epoch % 250 == 0 or epoch == (opt.niter-1):

            plt.imsave('%s/fake_sample_%s.png' %  (opt.outf, epoch), functions.convert_image_np(fake[0:1, :, :, :].detach()))
            plt.imsave('%s/fake_indicator_%s.png' %  (opt.outf, epoch), functions.convert_image_np(fake_ind[0:1, :, :, :].detach()))
            plt.imsave('%s/constraint_indicator_%s.png' %  (opt.outf, epoch), functions.convert_image_np(constraint_filled.detach()))
            plt.imsave('%s/background_%s.png' %  (opt.outf, epoch ), functions.convert_image_np(fake_background[0:1, :, :, :].detach()))
            plt.imsave('%s/fake_discriminator_heat_map_%s.png' %  (opt.outf, epoch), output[0, -1, :, :].detach().cpu().numpy())
            plt.imsave('%s/real_discriminator_heat_map_%s.png' %  (opt.outf, epoch), real_output[0, -1, :, :].detach().cpu().numpy())
            plt.imsave('%s/prev_%s.png' %  (opt.outf, epoch),functions.convert_image_np(prev[0:1, :, :, :].detach()))

    
            plt.plot(errD2plot)
            plt.savefig('%s/errD.png' %  (opt.outf))
            plt.close()
            plt.plot(errG2plot)
            plt.savefig('%s/errG.png' %  (opt.outf))
            plt.close()
            plt.plot(D_real2plot)
            plt.savefig('%s/error_D_real.png' %  (opt.outf))
            plt.close()
            plt.plot(D_fake2plot)
            plt.savefig('%s/error_D_fake.png' %  (opt.outf))
            plt.close()
            plt.plot(z_opt2plot)
            plt.savefig('%s/rec_loss.png' %  (opt.outf))
            plt.close()


 
            plt.imsave('%s/mask_down_%s.png' %  (opt.outf, epoch), mask_down[0, -1, :, :].detach().cpu().numpy(), cmap="gray")
            # plt.imsave('%s/const_down_%s.png' %  (opt.outf, epoch), const_down[0, -1, :, :].detach().cpu().numpy(), cmap="gray")
            # plt.imsave('%s/mask_mult_%s.png' %  (opt.outf, epoch), mask_mult[0, -1, :, :].detach().cpu().numpy())
            plt.imsave('%s/G(z_opt)_%s.png'    % (opt.outf, epoch),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()))
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))
        
        schedulerD.step()
        schedulerG.step()
        

    functions.save_networks(netG,netD,z_opt,opt)

    return z_opt,in_s,netG 


def init_models(opt):

    #generator initialization:
    
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG = nn.DataParallel(netG,device_ids=[0])
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD = nn.DataParallel(netD,device_ids=[0])
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    # print(netD)

    return netD, netG



def train_style(opt):
    torch.cuda.set_device(opt.device)
    
    img_size = (188, 255)
    prep = transforms.Compose([transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                    std=[1,1,1]),
                            transforms.Lambda(lambda x: x.mul_(255)),
                            ])
    
    prep_mask_source = transforms.Compose([
                            
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                    std=[1,1,1]),
                            transforms.Lambda(lambda x: x.mul_(255)),
                            ])
    
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                            transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                    std=[1,1,1]),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                            ])
    postpb = transforms.Compose([transforms.ToPILImage()])
    
    def postp(tensor): # to clip results in the range [0,1]
        t = postpa(tensor)
        t[t>1] = 1    
        t[t<0] = 0
        img = postpb(t)
        return img
    
    vgg = models.VGG()
    vgg.load_state_dict(torch.load('vgg_conv.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.cuda()
    
    
    
    # real_ = functions.read_image(opt)
    # real = imresize(real_,opt.scale1,opt)
    real = prep(Image.open('%s/%s' % (opt.input_dir,opt.input_name)))
    real = Variable(real.unsqueeze(0).cuda())
    

  
    

    # Loading masks and resizing so that biggest dimension is opt.patch_size 
    mask = functions.read_mask(opt,"Input/body_masks",opt.mask_name) 
    
     # Loading image source for mask and resizing so that biggest dimension is opt.patch_size 
    new_dim = (mask.size()[2], mask.size()[3])
    mask_source = Image.open('%s/%s' % ("Input/mask_sources",opt.mask_name[:-3]+"jpg"))
    mask_source = transforms.Resize(new_dim)(mask_source)
    mask_source = prep_mask_source(mask_source)
    mask_source = mask_source.unsqueeze(0).cuda()
   
   
    opt_img = Variable(real.data.clone(), requires_grad=True)


    constraint_ = functions.generate_eye_mask(opt, mask, 0, opt.fixed_eye_loc)
    # real = torch.ones_like(real)
    constraint = constraint_.to(mask_source) 
    
    style_layers = ['r11','r21','r31','r41', 'r51'] 
    loss_layers = style_layers
    loss_fns = [models.GramMSELoss()] * len(style_layers)
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        
    #these are good weights settings:
    style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
    weights = style_weights
    
    #compute optimization targets
    style_targets = [models.GramMatrix()(A).detach() for A in vgg(real, style_layers)]
    targets = style_targets

    #run style transfer
    max_iter = 500
    show_iter = 50
    # optimizer = torch.optim.Adam(vgg.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    optimizer = optim.LBFGS([opt_img]);

   
    n_iter=[0]

    while n_iter[0] <= max_iter:

        def closure():
            optimizer.zero_grad()
            fake, fake_ind, constraint_ind, mask_ind, constraint_filled = functions.gen_fake(real, opt_img, mask, constraint, mask_source, opt, no_constraint=True)

            # plt.imshow(postp(fake.data[0].cpu().squeeze()))
            # # plt.imshow((real/255).cpu().squeeze().permute(1,2,0).detach())
            # plt.show()
            out = vgg(fake, loss_layers)
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0]+=1
            #print loss
            if n_iter[0]%show_iter == (show_iter-1):
                print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
    #             print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
            return loss
        
        optimizer.step(closure)
        
    #display result
    # out_img = postp(opt_img.data[0].cpu().squeeze())
    # Making indicator for where the shape is for MTURK users
    

    dir2save = '%s/RandomSamples/%s/style/%s' % (opt.out, opt.input_name[:-4], opt.run_name)
    try:
        os.makedirs(dir2save + "/fake")
        os.makedirs(dir2save + "/background")
        os.makedirs(dir2save + "/mask")
        os.makedirs(dir2save + "/constraint")
        os.makedirs(dir2save + "/mask_ind")
        os.makedirs(dir2save + "/border_ind")

    except OSError:
        pass

    for i in range(20):
        
        fake, fake_ind, constraint_ind, mask_ind, constraint_filled = functions.gen_fake(real, opt_img, mask, constraint, mask_source, opt)
        dilated = torch.tensor(scipy.ndimage.morphology.binary_dilation(mask_ind,iterations=1)).to(mask_ind)
        border = (dilated-mask_ind).to(opt.device).round()*1
        border_colored= torch.zeros_like(fake)
        border_colored[:,-1:,:,:] = border*255 
        border_colored[:,:-1,:,:] = -border*255 
        border_ind = fake * (1-border) +border_colored
        
        fake = postp(fake.data[0].cpu().squeeze())
        real_ = postp(real.data[0].cpu().squeeze())
        border_ind = postp(border_ind.data[0].cpu().squeeze())
        # constraint_filled = postp(constraint_filled.data[0].cpu().squeeze())
     
        
        opt_img_ = postp(opt_img.data[0].cpu().squeeze())
        plt.imsave('%s/%s/%d.png' % (dir2save, "fake", i),  np.asarray(fake), vmin=0,vmax=1)
        # plt.imsave('%s/%s/%d.png' % (dir2save, "fake", i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
        plt.imsave('%s/%s/%d.png' % (dir2save, "background", i), np.asarray(opt_img_), vmin=0,vmax=1)
        # plt.imsave('%s/%s/%d.png' % (dir2save, "mask", i), np.asarray(constraint_filled), vmin=0,vmax=1)
        plt.imsave('%s/%s/%d.png' % (dir2save, "mask_ind", i), functions.convert_image_np(mask_ind.detach()),  cmap="gray")
        plt.imsave('%s/%s/%d.png' % (dir2save, "border_ind", i), np.asarray(border_ind))


        
    return





