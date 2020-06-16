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
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--crop_size', type=int, default=250)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--eye_diam', type=int, default=7)
    parser.add_argument('--eye_color', help='input eye color', default=(255, 255, 255))
    parser.add_argument('--border_width',type=int, default=4)
    parser.add_argument('--shade_amt', type=float, default=0.0)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--resize', action='store_true', help='enables random mask resize during training')
    parser.add_argument('--random_eye', action='store_true', help='enables random eye position during training')
    
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
    

    if (os.path.exists(dir2save)) and opt.run_name != "test":
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        #crop, _ , _ = functions.random_crop(real, opt.crop_size) 
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, crops, masks, eyes, NoiseAmp)

        im_height, im_width = reals[-1].size()[2], reals[-1].size()[3]
        #mask_locs = [(np.random.randint(im_height - opt.patch_size), np.random.randint(im_width - opt.patch_size)) for i in range(opt.num_samples)]
        mask_locs = [(63, 141), (60, 23), (74, 134), (82, 68), (90, 7), (29, 73), (44, 153), (77, 131), (29, 112), (10, 124), (83, 112), 
                   (10, 16), (72, 25), (17, 147), (84, 138), (61, 144), (12, 111), (81, 26), (20, 14), (19, 59)]
        SinGAN_generate(Gs,Zs,reals, crops, masks, eyes, NoiseAmp,opt, num_samples = opt.num_samples, mask_locs = mask_locs)
        random_crop_generate(reals[-1], masks[-1], eyes[-1], opt, num_samples = opt.num_samples, mask_locs = mask_locs)

