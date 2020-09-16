
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
    parser.add_argument('--mask_dir', help='input mask dir', default='Input/masks')
    parser.add_argument('--mask_name', help='input mask name', required=True)
    parser.add_argument('--mask_source', help='input mask source name', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--crop_size', type=int, default=250)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--mode', help='task to be done', default='style')
    parser.add_argument('--random_crop', action='store_true', help='enables random crop during training')
    parser.add_argument('--batch_size',type=int, default=1)
    
    
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
    

    if (os.path.exists(dir2save)) and opt.run_name != "test":
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        if opt.random_crop:
            real, _ , _ = functions.random_crop(real, opt.crop_size, opt) 
            
        # functions.show_image(real[0,:,:,:])
        functions.adjust_scales2image(real, opt)
        crop_sizes = [math.ceil((opt.crop_size*opt.scale1)*opt.scale_factor**(opt.stop_scale-i)) for i in range(opt.stop_scale + 1)]
        train(opt, Gs, Zs, reals, masks, constraints, crop_sizes, mask_sources, NoiseAmp)
        # SinGAN_generate(Gs,Zs,reals[:7], masks[:7], constraints[:7], crop_sizes[:7], mask_sources[:7], NoiseAmp[:7],opt, num_samples = opt.num_samples)
        SinGAN_generate(Gs,Zs,reals, masks, constraints, crop_sizes, mask_sources, NoiseAmp,opt, num_samples = opt.num_samples)
        # random_crop_generate(reals[6], masks[6], constraints[6], mask_sources[6], crop_sizes[6] ,opt, num_samples = opt.num_samples)
        random_crop_generate(reals[-1], masks[-1], constraints[-1], mask_sources[-1], crop_sizes[-1] ,opt, num_samples = opt.num_samples)
