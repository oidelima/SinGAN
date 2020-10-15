from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.models import *
from SinGAN.imresize import *
import SinGAN.functions as functions
import matplotlib.pyplot as plt
import random
from torchvision.utils import save_image

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--run_name', help='name of experimental run', required=True)
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mask_dir', help='input mask dir', default='Input/masks')
    parser.add_argument('--mask_name', help='input mask name', required=True)
    parser.add_argument('--mask_source', help='input mask source name', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--crop_size', type=int, default=250)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--random_crop', action='store_true', help='enables random crop during training')
    parser.add_argument('--batch_size',type=int, default=1)
    
    
    
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    dir2save = functions.generate_dir2save(opt)
    

    
    def load_ckpt(ckpt_name, models, optimizers=None):
        ckpt_dict = torch.load(ckpt_name)
        for prefix, model in models:
            assert isinstance(model, nn.Module)
            model.load_state_dict(ckpt_dict[prefix], strict=False)
        if optimizers is not None:
            for prefix, optimizer in optimizers:
                optimizer.load_state_dict(ckpt_dict[prefix])
        return ckpt_dict['n_iter']
    

    if (os.path.exists(dir2save)) and opt.run_name != "test":
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        LAMBDA_DICT = {
            'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
        
    
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        # torch.cuda.set_device(opt.device)

        real_ = functions.read_image(opt)
        real = imresize(real_,opt.scale1,opt)
        orig_size = (real.shape[2], real.shape[3])
        # real = imresize_to_shape(real_,(256, 256),opt)

        
        mask = functions.read_mask(opt,"Input/body_masks",opt.mask_name) 
        # constraint = functions.read_mask(opt, "Input/constraints", opt.mask_name)
        # mask_ = functions.read_mask(opt) 
        # mask_ = imresize_mask_to_shape(mask_, (256, 256), opt)

       # Loading image source for mask and resizing so that biggest dimension is opt.patch_size 
        new_dim = (mask.size()[3], mask.size()[2])
        mask_source = functions.read_image(opt, "Input/mask_sources", opt.mask_name[:-3]+"jpg", size=new_dim)
        
        
        constraint_ = functions.generate_eye_mask(opt, mask, 0)
        constraint = constraint_ * mask #* mask_source
        
        
        height, width = real.shape[2], real.shape[3]
        scale = 256/min(height, width)
        size = (int(height*scale), int(width*scale))
  
        
        img_transform = transforms.Compose(
            [   transforms.Resize(size),
                transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)])
        mask_transform = transforms.Compose(
            [ transforms.Resize(size=(mask.shape[2], mask.shape[3]), interpolation=1),transforms.ToTensor()])
        
       

        model = PConvUNet().to(opt.device)
        load_ckpt("weights.pth", [('model', model)])    
        model.eval()


        real = Image.open("Input\Images\\" + opt.input_name)
        real_ = img_transform(real).unsqueeze(0).to(opt.device)
        

        real_cropped = real_[:,:,int((size[0] - 256)/2):int((size[0] + 257)/2), int((size[1] - 256)/2): int((size[1] + 257)/2)]

        
        
        mask = Image.open("Input\\body_masks\\" + opt.mask_name)
        mask_ = mask_transform(mask).unsqueeze(0).to(opt.device)
      
        for i in range(20):
        
            _, _, constraint_ind, mask_ind, constraint_filled = functions.gen_fake(real_cropped, real_cropped, mask_[:,0:1,:,:], constraint_, mask_source, opt)
            mask = 1 - mask_ind #* (1-constraint_ind)
            mask = mask.repeat(1,3,1,1)
            real = real_cropped#*(1-constraint_ind).to(opt.device) + (constraint_ind).to(opt.device)

            with torch.no_grad():
                out, out_mask = model(real*mask.to(opt.device), mask.to(opt.device))
        
            output_comp = (1 - mask.to(opt.device)) * out.to(opt.device) + ((mask-constraint_ind).to(opt.device) * real).to(opt.device) + constraint_filled.to(opt.device)
            
            dilated = torch.tensor(scipy.ndimage.morphology.binary_dilation(mask_ind,iterations=1)).to(mask_ind)
            border = (dilated-mask_ind).to(opt.device).round()
            border_colored= torch.zeros_like(real)
            border_colored[:,0:1,:,:] = border*255 
            border_colored[:,1:,:,:] = -border*255 
            border_ind = output_comp * (1-border) +border_colored
            
            
            
            output_full = real_.clone()
            border_ind_full =  real_.clone()
            mask_ind_full = torch.zeros_like(real_)
            output_full[:,:,int((size[0] - 256)/2):int((size[0] + 257)/2), int((size[1] - 256)/2): int((size[1] + 257)/2)]  =output_comp
            border_ind_full[:,:,int((size[0] - 256)/2):int((size[0] + 257)/2), int((size[1] - 256)/2): int((size[1] + 257)/2)]  =border_ind
            mask_ind_full[:,:,int((size[0] - 256)/2):int((size[0] + 257)/2), int((size[1] - 256)/2): int((size[1] + 257)/2)]  =mask_ind
            
            
            def post(image):
                image = image.transpose(1, 3)
                image = image * torch.tensor(STD).to(opt.device) + torch.tensor(MEAN).to(opt.device)
                image = image.transpose(1, 3)
                image = torch.nn.functional.interpolate(image, size=orig_size, mode='nearest')
                return image
        
            output_full = post(output_full)
            mask_ind_full = torch.nn.functional.interpolate(mask_ind_full, size=orig_size, mode='nearest')
            border_ind_full= post(border_ind_full)
           
            
           

            
            dir2save = '%s/RandomSamples/%s/NVIDIA/%s' % (opt.out, opt.input_name[:-4], opt.run_name)
            
            try:
                os.makedirs(dir2save + "/fake")
                os.makedirs(dir2save + "/border_ind")
                # os.makedirs(dir2save + "/background")
                # os.makedirs(dir2save + "/mask")
                # os.makedirs(dir2save + "/constraint")
                os.makedirs(dir2save + "/mask_ind")

            except OSError:
                pass


            save_image(output_full, '%s/%s/%d.png' % (dir2save, "fake", i))
            save_image(border_ind_full, '%s/%s/%d.png' % (dir2save, "border_ind", i))
            save_image(mask_ind_full, '%s/%s/%d.png' % (dir2save, "mask_ind", i))
            # plt.imsave('%s/%s/%d.png' % (dir2save, "fake", i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
            # plt.imsave('%s/%s/%d.png' % (dir2save, "background", i), np.asarray(opt_img_), vmin=0,vmax=1)
            # plt.imsave('%s/%s/%d.png' % (dir2save, "mask", i), functions.convert_image_np(fake_ind.detach()), vmin=0,vmax=1)
            # plt.imsave('%s/%s/%d.png' % (dir2save, "mask_ind", i), functions.convert_image_np(mask_ind.detach()),  cmap="gray")
            # plt.imsave('%s/%s/%d.png' % (dir2save, "constraint", i), functions.convert_image_np(constraint_filled.detach()), vmin=0,vmax=1)

                    
                
        

        
        
        
    
