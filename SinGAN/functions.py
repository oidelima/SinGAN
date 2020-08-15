import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
from SinGAN.imresize import imresize, imresize_mask, batch_imresize
import os
import random
from sklearn.cluster import KMeans
import scipy
from PIL import Image, ImageDraw



# custom weights initialization called on netG and netD

def read_image(opt):
    x = img.imread('%s%s' % (opt.input_img,opt.ref_image))
    return np2torch(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def make_binary(mask, opt):
    mask[mask>opt.mask_epsilon] = 1
    mask[mask<=opt.mask_epsilon] = 0
    return mask

#def denorm2image(I1,I2):
#    out = (I1-I1.mean())/(I1.max()-I1.min())
#    out = out*(I2.max()-I2.min())+I2.mean()
#    return out#.clamp(I2.min(), I2.max())

#def norm2image(I1,I2):
#    out = (I1-I2.mean())*2
#    return out#.clamp(I2.min(), I2.max())

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp

def generate_noise(size,num_samp=1,device='cuda:0',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t, opt):
    if (torch.cuda.is_available()):
        t = t.to(opt.device)
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)


    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image(opt, input_dir=None, input_name=None):
    if input_dir and input_name: x = img.imread('%s/%s' % (input_dir,input_name))
    else: x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def random_crop(real, crop_size, opt):
    
    height = real.shape[2]
    width = real.shape[3]
    crop = torch.zeros((opt.batch_size, 3, crop_size, crop_size))
     
    for i in range(opt.batch_size):              
        h_idx = np.random.randint(height - crop_size, size=1)
        w_idx = np.random.randint(width - crop_size, size=1)
        h_idx_end = h_idx + crop_size
        w_idx_end = w_idx + crop_size
        crop[i] = real[:,:,h_idx[0]:h_idx_end[0],w_idx[0]:w_idx_end[0]]
        if i == 0: 
            h_idx_ = h_idx
            w_idx_ = w_idx

    return crop.to(opt.device), int(h_idx_), int(w_idx_)

def read_mask(opt, mask_dir=None, mask_name=None):
    #x = img.imread('%s/%s' % (opt.mask_dir,opt.mask_name))
    if mask_dir and mask_name: x = Image.open('%s/%s' % (mask_dir,mask_name))
    else: x = Image.open('%s/%s' % (opt.mask_dir,opt.mask_name))
    x = np.array(x)
    x = preprocess_mask(x, opt) 
    x = x[:,:,None,None]
    x = x.transpose((3, 2, 0, 1))
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x, opt)
    #x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    return x

# def generate_eye_mask(opt, mask, level):
    
#     scale = math.pow(opt.scale_factor, level)
    
#     im_height, im_width = mask.size()[2], mask.size()[3]
#     # Make eye constraint mask
#     eye_diam = opt.eye_diam
#     eye = Image.new('RGB', (mask.size()[2], mask.size()[3]))
#     draw = ImageDraw.Draw(eye)
#     eye_loc = find_valid_eye_location(opt, eye_diam, mask)
#     eye_loc = (85, 133) #TODO

#     draw.ellipse([(eye_loc[1], eye_loc[0]), (eye_loc[1] + eye_diam, eye_loc[0] + eye_diam)], fill="white")
#     eye = torch.from_numpy(np.array(eye)).permute((2, 0, 1))
#     eye[eye>0] = 1 
    
    
#     if not(opt.not_cuda):
#         eye = move_to_gpu(eye)
#         eye = eye.unsqueeze(0)
        
#     eye = imresize_mask(eye,scale,opt)
    
#     return eye

        
# def find_valid_eye_location(opt, eye_diam, mask):

#     while True:
#         try:
#             loc = (random.randint(0, mask.size()[2]), random.randint(0, mask.size()[3]))   
#             if mask[:, :, loc[0], loc[1]] == 1 and mask[:, :, loc[0] + eye_diam, loc[1] + eye_diam] == 1:
#                 return loc
#         except:
#             pass

def gen_fake(real, fake_background, mask, constraint, mask_source, opt):
       
    im_height, im_width = real.size()[2], real.size()[3] 
    mask_height, mask_width = mask.size()[2], mask.size()[3] 
    
    fake = real.clone()
    fake_ind = torch.zeros((opt.batch_size, 3, im_height, im_width))
    mask_ind = torch.zeros((opt.batch_size, 1, im_height, im_width))
    constraint_ind = torch.zeros((opt.batch_size, 3, im_height, im_width))


    for i in range(opt.batch_size):
        
        
        h_loc = np.random.randint(im_height - mask_height)
        w_loc = np.random.randint(im_width - mask_width)

        
        fake[i,:,h_loc:h_loc + mask_height ,w_loc:w_loc + mask_width] = fake_background[i,:,0:mask_height ,0:mask_width] *(mask)*(1-constraint) \
                                                                            + real[i,:,h_loc:h_loc+mask_height ,w_loc:w_loc +mask_width]*(1-mask) \
                                                                            + constraint.to(opt.device)*mask_source 
        
 
        fake_ind[i,:,h_loc:h_loc+mask_height ,w_loc:w_loc + mask_width] =  fake_background[i,:,0:mask_height ,0:mask_width] *(mask) 
        mask_ind[i,:,h_loc:h_loc+mask_height ,w_loc:w_loc + mask_width] =  mask
        constraint_ind[i,:,h_loc:h_loc+mask_height ,w_loc:w_loc + mask_width] = constraint

        
            
    return fake, fake_ind, constraint_ind, mask_ind
 
# def get_eye_color(real):
#     height, width = real.size()[2], real.size()[3]
#     eye_color_loc = (random.randint(0, height-1), random.randint(0, width-1))
#     eye_color =  real[0, :, eye_color_loc[0], eye_color_loc[1]]
#     eye_color = 255*denorm(eye_color)
#     return eye_color
    
                                                                                
def pad_mask(mask, input_size):
    mask_height, mask_width = mask.size()[2], mask.size()[3] 
    pad_down = input_size[0] - mask_height
    pad_right = input_size[1] - mask_width
    p2d = (0, pad_right, 0, pad_down) 
    mask = torch.nn.functional.pad(mask, p2d, "constant", 0)
    return mask

# def make_input(noise, mask, eye, opt):
#     noise_height, noise_width = noise.size()[2], noise.size()[3] 
#     mask = mask.repeat(opt.batch_size, 1, 1, 1)
#     mask = pad_mask(mask, (noise_height, noise_width)).float() # Padding masks to make same size as input 
#     # eye_in = pad_mask(eye[:, 0:1, :, :], (noise_height, noise_width)).float()
#     eye_in = pad_mask(eye[:, :, :, :], (noise_height, noise_width)).float().repeat(opt.batch_size, 1, 1, 1)
    
#     eye_colored = eye_in.clone()
#     eye_colored[:, 0, :, :] *= (opt.eye_color[0]/255)
#     eye_colored[:, 1, :, :] *= (opt.eye_color[1]/255)
#     eye_colored[:, 2, :, :] *= (opt.eye_color[2]/255)
#     # plt.imsave('eye_test', eye_colored[0, -1, :, :].detach().cpu().numpy())

#     #G_input = torch.cat((noise, mask, eye_colored), dim=1) # concatenating to make input to generator
#     G_input = noise
#     return G_input                 

def preprocess_mask(im, opt):
    # Pads mask to make it square and then resizes
    h = im.shape[0]
    w = im.shape[1]

    if h > w:
        diff = h - w
        if diff%2 == 0:
            new_img = np.pad(im, ((0, 0), (diff // 2, diff // 2)), mode='constant')
        else:
            new_img = np.pad(im, ((0, 0), (diff // 2, diff // 2 + 1)), mode='constant')
    elif w > h:
        diff = w - h
        if diff%2 == 0:
            new_img = np.pad(im, ((diff // 2, diff // 2), (0, 0)), mode='constant')
        else:
            new_img = np.pad(im, ((diff // 2, diff // 2 + 1), (0, 0)), mode='constant')
    else: # already square
        new_img = im
    # fill in small holes
    new_img = scipy.ndimage.morphology.binary_dilation(new_img, iterations=4).astype(np.uint8)
    pil_im = Image.fromarray(new_img * 255.0)
    pil_im.thumbnail((opt.patch_size, opt.patch_size), Image.NEAREST)
    return np.array(pil_im) / 255.0

    
def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def np2torch(x,opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255   
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x, opt)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    x = norm(x)
    
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netD,z,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_,opt):
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size , max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real


def create_pyramid(im,pyr_list,opt, mode=None):

    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        if mode == "mask":
            curr_im = imresize_mask(im,scale,opt)  
        else:        
            curr_im = imresize(im,scale,opt)
        pyr_list.append(curr_im.to(opt.device))
    return pyr_list


def load_trained_pyramid(opt, mode_='train'):
    #dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'TrainedModels/%s/name=%s, scale_factor=%f,alpha=%d' % (opt.input_name[:-4],opt.run_name, opt.scale_factor_init,opt.alpha)
    elif (opt.mode == 'animation_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:"+ str(opt.gpu))
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num

def quant(prev,device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x, opt)
    x = x.type(torch.cuda.FloatTensor) if () else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor.to(device))
    x = x.view(prev.shape)
    return x,centers

def quant2centers(paint, centers):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x, opt)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    #x = x.type(torch.cuda.FloatTensor)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask,opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0,vmax=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask


def draw_concat(Gs,Zs, reals, crops, masks, constraints, NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s

    if opt.random_crop:
        reals = crops # use crop sizes if cropping

    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next, mask_curr, constraint_curr, noise_amp in zip(Gs,Zs,reals,reals[1:], masks, constraints, NoiseAmp):
                if count == 0:
                    z = generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device,num_samp=opt.batch_size)
                    z = z.expand(opt.batch_size, 3, z.shape[2], z.shape[3])
                else:
                    z = generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device, num_samp=opt.batch_size)
                z = m_noise(z)
                
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z).to(opt.device)

                z_in = noise_amp*z+G_z
                # G_input = make_input(z_in, mask_curr, eye_curr, opt)
                G_z = G(z_in.detach(),G_z)
                G_z = batch_imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,mask_curr, constraint_curr, noise_amp in zip(Gs,Zs,reals,reals[1:], masks, constraints, NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                
                G_z = m_image(G_z).to(opt.device)
                
                z_in = noise_amp*Z_opt+G_z
                # G_input = make_input(z_in, mask_curr, eye_curr, opt)
                G_z = G(z_in.detach(),G_z)
                G_z = batch_imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1

    return G_z.to(opt.device)


