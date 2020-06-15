from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--run_name', help='name of experimental run', required=True)
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mask_dir', help='input mask dir', default='Input/masks')
    parser.add_argument('--mask_name', help='input mask name', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--crop_size', type=int, default=250)
    parser.add_argument('--eye_diam', type=int, default=7)
    parser.add_argument('--eye_color', help='input eye color', default=(255, 255, 255))
    parser.add_argument('--border_width',type=int, default=4)
    parser.add_argument('--shade_amt', type=float, default=0.0)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    masks = []
    eyes = []
    crops = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        #crop, _ , _ = functions.random_crop(real, opt.crop_size) 
        #functions.adjust_scales2image(crop, opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, crops, masks, eyes, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals, crops, masks, eyes, NoiseAmp,opt)
        random_crop_generate(reals[-1], masks[-1], eyes[-1], opt)
