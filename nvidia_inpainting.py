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
        # real = imresize_to_shape(real_,(256, 256),opt)

        
        mask_ = functions.read_mask(opt) 
        # mask_ = imresize_mask_to_shape(mask_, (256, 256), opt)

        
        constraint = functions.read_mask(opt, "Input/custom_constraints", opt.mask_name) 
        mask_source = functions.read_image(opt, "Input/mask_sources", opt.mask_source)
        mask_source = nn.functional.interpolate(mask_source, size=(opt.patch_size, opt.patch_size))
        constraint_ = constraint * mask_ #* mask_source
        
        #test eye
        mask_source = torch.ones_like(mask_source)

        #ocean
        opt.eye_diam=4
        opt.eye_loc = (38, 78) #TODO ocean
        mask_source[:,0,:,:]  = (241/255 - 0.485)/0.229
        mask_source[:,1,:,:]  = (238/255 - 0.456)/0.224
        mask_source[:,2,:,:]  = (240/255 - 0.406)/0.225

        #tetra_fish
        # opt.eye_diam = 4
        # opt.eye_loc = (40, 75) #TODO 
        # mask_source[:,0,:,:]  = (148/255 - 0.5)*2
        # mask_source[:,1,:,:]  = (151/255 - 0.5)*2
        # mask_source[:,2,:,:]  = (124/255 - 0.5)*2

        #blackbird
        # opt.eye_diam = 4
        # opt.eye_loc = (28, 43) #TODO 
        # mask_source[:,0,:,:]  = (255/255 - 0.5)*2
        # mask_source[:,1,:,:]  = (231/255 - 0.5)*2
        # mask_source[:,2,:,:]  = (184/255 - 0.5)*2

        # rabbit
        # opt.eye_diam = 4
        # opt.eye_loc = (60, 43) #TODO 
        # mask_source[:,0,:,:]  = (168/255 - 0.5)*2
        # mask_source[:,1,:,:]  = (176/255 - 0.5)*2
        # mask_source[:,2,:,:]  = (155/255 - 0.5)*2

        
        constraint_ = functions.generate_eye_mask(opt, mask_, 0)
        
        
        # _, _, constraint_ind, mask_ind, _= functions.gen_fake(real, real, mask_.to(real), constraint_.to(real), mask_source.to(real), opt)
         

        # mask_without_eye = mask_ind * (1-constraint_ind)


        size = (256, 256)
        img_transform = transforms.Compose(
            [transforms.Resize(size=size), transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)])
        mask_transform = transforms.Compose(
            [ transforms.Resize(size=(96, 96), interpolation=1),transforms.ToTensor()])
        
       

        model = PConvUNet().to(opt.device)
        load_ckpt("weights.pth", [('model', model)])    
        model.eval()


        real = Image.open("Input\Images\\181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08.jpg")
        real_ = img_transform(real).unsqueeze(0).to(opt.device)

        
        mask = Image.open("Input\masks\\bird-20.gif")
        mask_ = mask_transform(mask).unsqueeze(0).to(opt.device)
        
        
        for i in range(20):
        
            _, _, constraint_ind, mask_ind, constraint_filled = functions.gen_fake(real_, real_, mask_[:,0:1,:,:], constraint_, mask_source, opt)
            mask = 1 - mask_ind * (1-constraint_ind)
            # mask = mask.repeat(1,3,1,1)
            real = real_*(1-constraint_ind).to(opt.device) + (constraint_filled).to(opt.device)
            # real = real_
            # model.eval()
            with torch.no_grad():
                out, out_mask = model(real*mask.to(opt.device), mask.to(opt.device))
        
            
            output_comp = (1 - mask.to(opt.device)) * out.to(opt.device) + (mask.to(opt.device) * real).to(opt.device)
            
            output_comp = output_comp.transpose(1, 3)
            output_comp = output_comp * torch.tensor(STD).to(opt.device) + torch.tensor(MEAN).to(opt.device)
            output_comp = output_comp.transpose(1, 3)
            output_comp = torch.nn.functional.interpolate(output_comp, size=(188, 250))
            
            # out = out.transpose(1, 3)
            # out= out * torch.tensor(STD).to(opt.device) + torch.tensor(MEAN).to(opt.device)
            # out = out.transpose(1, 3)
            
            dir2save = '%s/RandomSamples/%s/NVIDIA/%s' % (opt.out, opt.input_name[:-4], opt.run_name)
            
            try:
                os.makedirs(dir2save + "/fake")
                # os.makedirs(dir2save + "/background")
                # os.makedirs(dir2save + "/mask")
                # os.makedirs(dir2save + "/constraint")
                # os.makedirs(dir2save + "/mask_ind")

            except OSError:
                pass


            save_image(output_comp, '%s/%s/%d.png' % (dir2save, "fake", i))
            # plt.imsave('%s/%s/%d.png' % (dir2save, "fake", i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
            # plt.imsave('%s/%s/%d.png' % (dir2save, "background", i), np.asarray(opt_img_), vmin=0,vmax=1)
            # plt.imsave('%s/%s/%d.png' % (dir2save, "mask", i), functions.convert_image_np(fake_ind.detach()), vmin=0,vmax=1)
            # plt.imsave('%s/%s/%d.png' % (dir2save, "mask_ind", i), functions.convert_image_np(mask_ind.detach()),  cmap="gray")
            # plt.imsave('%s/%s/%d.png' % (dir2save, "constraint", i), functions.convert_image_np(constraint_filled.detach()), vmin=0,vmax=1)

                    
                
        

        
        
        
    
