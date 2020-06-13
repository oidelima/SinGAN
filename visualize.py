import visdom
import imageio
import numpy as np
import sys, os

viz = visdom.Visdom()
dir = os.path.join('Output', 'RandomSamples')

NUM_EXAMPLES = 20

curr_img = 0
img = "94547580-ivy-growing-on-the-forest-floor"
part = "fake"
method_1 = "random_crop_"
method_2 = "SinGAN_"

images_window = viz.images(np.zeros((2, 3, 250, 250)))

def type_callback(event):
    global curr_img, images_window, viz
    if event['event_type'] == 'KeyPress':
        #curr_img = event['pane_data']['i']
        if event['key'] == 'Enter' and curr_img < NUM_EXAMPLES:
            gan_img = imageio.imread(os.path.join(dir, img, method_2, part, str(curr_img)+".png"))[:, :, :3]
            gan_img = np.transpose(gan_img, (2, 0, 1)) 
            gan_img = np.expand_dims(gan_img, axis = 0)  
            crop_img = imageio.imread(os.path.join(dir, img, method_1, part, str(curr_img)+".png"))[:, :, :3]
            crop_img = np.transpose(crop_img, (2, 0, 1)) 
            crop_img = np.expand_dims(crop_img, axis = 0)
            joint_img = np.concatenate((gan_img, crop_img), axis=0)
            images_window = viz.images(joint_img, win = images_window, opts={'title': 'SinGAN vs Random Crop. Example: {} / {}'.format(curr_img + 1, NUM_EXAMPLES),
                                                                             'height':10000})
            viz.update_window_opts(
                win=images_window,
                opts=dict(
                    width=1000,
                    height=1000,
                ),
            )
            
            curr_img += 1
        
viz.register_event_handler(type_callback, images_window)  
input('Waiting for callbacks, press enter to quit.')       
            



# for img in os.listdir(dir):
#     if img in images:
#         for method in os.listdir(os.path.join(dir, img)):
#             if method in methods:
#                 for part in os.listdir(os.path.join(dir, img, method)):
#                     if part in parts:
#                         for i in os.listdir(os.path.join(dir, img, method, part)):
                            
#                             file = imageio.imread(os.path.join(dir, img, method, part, i)) 
#                             file = file[:, :, :3]
#                             file = np.transpose(file, (2, 0, 1))  
#                             print("Img: {}, Method: {}, Part: {}, Number {} sent to server".format(img, method, part, i))
                        
        #print(filename)

