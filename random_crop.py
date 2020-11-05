
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import matplotlib.pyplot as plt
import random



if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--run_name', help='name of experimental run', required=True)
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mask_dir', help='input mask dir', default='Input/body_masks')
    parser.add_argument('--mask_name', help='input mask name', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--crop_size', type=int, default=250)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--mode', help='task to be done', default='style')
    parser.add_argument('--random_crop', action='store_true', help='enables random crop during training')
    parser.add_argument('--batch_size',type=int, default=1)
    parser.add_argument('--fixed_eye_loc', nargs='+', type=int)
    

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    masks = []
    constraints = []
    mask_sources = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    
    torch.cuda.set_device(opt.device)

    if (os.path.exists(dir2save)) and opt.run_name != "test":
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        
        # functions.show_image(real[0,:,:,:])
        functions.adjust_scales2image(real, opt)
        
         # Loading input background image, normalizing and resizing to largest scale size
        real = functions.read_image(opt)
        real = imresize(real,opt.scale1,opt)

        # Loading masks and resizing so that biggest dimension is opt.patch_size 
        mask = functions.read_mask(opt,"Input/body_masks",opt.mask_name) 
        # constraint = functions.read_mask(opt, "Input/constraints", opt.mask_name) 
        
        # Loading image source for mask and resizing so that biggest dimension is opt.patch_size 
        new_dim = (mask.size()[3], mask.size()[2])
        mask_source = functions.read_image(opt, "Input/mask_sources", opt.mask_name[:-3]+"jpg", size=new_dim)

        constraint_ = functions.generate_eye_mask(opt, mask, 0, opt.fixed_eye_loc)
        constraint = constraint_ * mask #* mask_source

        random_crop_generate(real, mask, constraint, mask_source ,opt, num_samples = opt.num_samples)
