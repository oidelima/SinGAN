
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
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
    parser.add_argument('--eye_diam', type=int, default=9)
    parser.add_argument('--eye_color', help='input eye color', default=(255, 255, 255))
    parser.add_argument('--random_eye', action='store_true', help='enables random eye position during training')
    parser.add_argument('--random_eye_color', action='store_true', help='enables random eye color during training')
    parser.add_argument('--border_width',type=int, default=0)
    parser.add_argument('--shade_amt', type=float, default=0.0)
    parser.add_argument('--mask_epsilon', type=float, default=0.01)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--resize', action='store_true', help='enables random mask resize during training')
    parser.add_argument('--random_crop', action='store_true', help='enables random crop during training')
    parser.add_argument('--upweight', action='store_true', help='enables random crop during training')
    parser.add_argument('--batch_size',type=int, default=5)
    
    
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    torch.cuda.set_device(opt.device)
    Gs = []
    Zs = []
    reals = []
    masks = []
    crops = []
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
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, crops, masks, constraints,mask_sources, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals[:7], crops[:7], masks[:7], constraints[:7], mask_sources[:7], NoiseAmp[:7],opt, num_samples = opt.num_samples)
        random_crop_generate(reals[6], masks[6], constraints[6], mask_sources[6], crops[6], opt, num_samples = opt.num_samples)

