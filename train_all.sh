# tmux new -d -s 0
tmux new -d -s 1
tmux new -d -s 2
# tmux new -d -s 3
# tmux new -d -s 4
# tmux new -d -s 5
#tmux new -d -s 6
# tmux new -d -s 7
# tmux new -d -s 8
# tmux new -d -s 9


# #SinGAN

# tmux send-keys -t "1" "python main_train.py --input_name 78308399-texture-of-red-gravel-red-rocky-floors-a-wall-of-red-gravel-stones-small-and-medium-sized-sharp-edge.jpg --mask_name=bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 25 52"   Enter
# tmux send-keys -t "1" "python main_train.py --input_name CF005_L_V_2012-c.jpg --mask_name=bird-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 41 68" Enter
# tmux send-keys -t "1" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name=bird.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 29 33" Enter
# tmux send-keys -t "1" "python main_train.py --input_name 42482737980_94dcb940be_k.jpg --mask_name=butterfly.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000  --mode=train --fixed_eye_loc 19 58" Enter
# tmux send-keys -t "1" "python main_train.py --input_name dan-woje-desertmix-surfacev2.jpg --mask_name=camel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 28 55" Enter
# tmux send-keys -t "1" "python main_train.py --input_name CP030_L_V_2013-c.jpg --mask_name=cat-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 46 22" Enter
# tmux send-keys -t "1" "python main_train.py --input_name walden-brush-view6.jpg --mask_name=cat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 35 82" Enter
# tmux send-keys -t "1" "python main_train.py --input_name savanna_1.jpg --mask_name=cheetah.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=train --fixed_eye_loc 37 45" Enter
# tmux send-keys -t "1" "python main_train.py --input_name walden-brush2-view1.jpg --mask_name=cow.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 30 12" Enter
# tmux send-keys -t "1" "python main_train.py --input_name charlottesville-3-view15.jpg --mask_name=dog.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 47 83" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 14005790586_349d8028b5_k.jpg --mask_name=dolphin.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 31 55" Enter
# tmux send-keys -t "2" "python main_train.py --input_name charlottesville-7-view1.jpg --mask_name=deer.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 45 61" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 27186282509_126fb93851_k.jpg --mask_name=eagle.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 38 81" Enter
# tmux send-keys -t "2" "python main_train.py --input_name ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea.jpg --mask_name=elephant.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=train --fixed_eye_loc 47 40" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 3380728556_14574fff67_k.jpg --mask_name=fish-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 54 88" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 1388693307.jpg --mask_name=fish-3.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 79 56" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 65533488.jpg --mask_name=fish-4.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 28 43" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 247770075.jpg --mask_name=fish.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 20 71" Enter
# tmux send-keys -t "2" "python main_train.py --input_name charlottesville-7-view.jpg --mask_name=frog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 37 59" Enter
# tmux send-keys -t "2" "python main_train.py --input_name walen-log-view1.jpg --mask_name=goat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 52 73" Enter


# #SinGAN Inpainting 

# tmux send-keys -t "1" "python main_train.py --input_name 78308399-texture-of-red-gravel-red-rocky-floors-a-wall-of-red-gravel-stones-small-and-medium-sized-sharp-edge.jpg --mask_name=bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting"   Enter
# tmux send-keys -t "1" "python main_train.py --input_name CF005_L_V_2012-c.jpg --mask_name=bird-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting " Enter
# tmux send-keys -t "1" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name=bird.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "1" "python main_train.py --input_name 42482737980_94dcb940be_k.jpg --mask_name=butterfly.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000  --mode=inpainting" Enter
# tmux send-keys -t "1" "python main_train.py --input_name dan-woje-desertmix-surfacev2.jpg --mask_name=camel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "1" "python main_train.py --input_name CP030_L_V_2013-c.jpg --mask_name=cat-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "1" "python main_train.py --input_name walden-brush-view6.jpg --mask_name=cat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "1" "python main_train.py --input_name savanna_1.jpg --mask_name=cheetah.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=inpainting" Enter
# tmux send-keys -t "1" "python main_train.py --input_name walden-brush2-view1.jpg --mask_name=cow.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "1" "python main_train.py --input_name charlottesville-3-view15.jpg --mask_name=dog.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 14005790586_349d8028b5_k.jpg --mask_name=dolphin.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "2" "python main_train.py --input_name charlottesville-7-view1.jpg --mask_name=deer.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 27186282509_126fb93851_k.jpg --mask_name=eagle.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting " Enter
# tmux send-keys -t "2" "python main_train.py --input_name ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea.jpg --mask_name=elephant.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=inpainting" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 3380728556_14574fff67_k.jpg --mask_name=fish-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 1388693307.jpg --mask_name=fish-3.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 65533488.jpg --mask_name=fish-4.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 247770075.jpg --mask_name=fish.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting " Enter
# tmux send-keys -t "2" "python main_train.py --input_name charlottesville-7-view.jpg --mask_name=frog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter
# tmux send-keys -t "2" "python main_train.py --input_name walen-log-view1.jpg --mask_name=goat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting" Enter

#Random Crop


# tmux send-keys -t "1" "python random_crop.py --input_name 78308399-texture-of-red-gravel-red-rocky-floors-a-wall-of-red-gravel-stones-small-and-medium-sized-sharp-edge.jpg --mask_name=bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 25 52"   Enter
# tmux send-keys -t "1" "python random_crop.py --input_name CF005_L_V_2012-c.jpg --mask_name=bird-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 41 68" Enter
# tmux send-keys -t "1" "python random_crop.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name=bird.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 29 33" Enter
# tmux send-keys -t "1" "python random_crop.py --input_name 42482737980_94dcb940be_k.jpg --mask_name=butterfly.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000  --mode=train --fixed_eye_loc 19 58" Enter
# tmux send-keys -t "1" "python random_crop.py --input_name dan-woje-desertmix-surfacev2.jpg --mask_name=camel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 28 55" Enter
# tmux send-keys -t "1" "python random_crop.py --input_name CP030_L_V_2013-c.jpg --mask_name=cat-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 46 22" Enter
# tmux send-keys -t "1" "python random_crop.py --input_name walden-brush-view6.jpg --mask_name=cat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 35 82" Enter
# tmux send-keys -t "1" "python random_crop.py --input_name savanna_1.jpg --mask_name=cheetah.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=train --fixed_eye_loc 37 45" Enter
# tmux send-keys -t "1" "python random_crop.py --input_name walden-brush2-view1.jpg --mask_name=cow.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 30 12" Enter
# tmux send-keys -t "1" "python random_crop.py --input_name charlottesville-3-view15.jpg --mask_name=dog.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 47 83" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name 14005790586_349d8028b5_k.jpg --mask_name=dolphin.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 31 55" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name charlottesville-7-view1.jpg --mask_name=deer.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 45 61" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name 27186282509_126fb93851_k.jpg --mask_name=eagle.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 38 81" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea.jpg --mask_name=elephant.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=train --fixed_eye_loc 47 40" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name 3380728556_14574fff67_k.jpg --mask_name=fish-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 54 88" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name 1388693307.jpg --mask_name=fish-3.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 79 56" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name 65533488.jpg --mask_name=fish-4.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 28 43" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name 247770075.jpg --mask_name=fish.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train 20 71" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name charlottesville-7-view.jpg --mask_name=frog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 37 59" Enter
# tmux send-keys -t "2" "python random_crop.py --input_name walen-log-view1.jpg --mask_name=goat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 52 73" Enter


# Style transfer

# tmux send-keys -t "1" "python style.py --input_name 78308399-texture-of-red-gravel-red-rocky-floors-a-wall-of-red-gravel-stones-small-and-medium-sized-sharp-edge.jpg --mask_name=bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style"   Enter
# tmux send-keys -t "1" "python style.py --input_name CF005_L_V_2012-c.jpg --mask_name=bird-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style " Enter
# tmux send-keys -t "1" "python style.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name=bird.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "1" "python style.py --input_name 42482737980_94dcb940be_k.jpg --mask_name=butterfly.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000  --mode=style" Enter
# tmux send-keys -t "1" "python style.py --input_name dan-woje-desertmix-surfacev2.jpg --mask_name=camel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "1" "python style.py --input_name CP030_L_V_2013-c.jpg --mask_name=cat-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "1" "python style.py --input_name walden-brush-view6.jpg --mask_name=cat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "1" "python style.py --input_name savanna_1.jpg --mask_name=cheetah.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=style" Enter
# tmux send-keys -t "1" "python style.py --input_name walden-brush2-view1.jpg --mask_name=cow.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "1" "python style.py --input_name charlottesville-3-view15.jpg --mask_name=dog.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "2" "python style.py --input_name 14005790586_349d8028b5_k.jpg --mask_name=dolphin.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "2" "python style.py --input_name charlottesville-7-view1.jpg --mask_name=deer.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "2" "python style.py --input_name 27186282509_126fb93851_k.jpg --mask_name=eagle.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style " Enter
# tmux send-keys -t "2" "python style.py --input_name ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea.jpg --mask_name=elephant.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=style" Enter
# tmux send-keys -t "2" "python style.py --input_name 3380728556_14574fff67_k.jpg --mask_name=fish-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "2" "python style.py --input_name 1388693307.jpg --mask_name=fish-3.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "2" "python style.py --input_name 65533488.jpg --mask_name=fish-4.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "2" "python style.py --input_name 247770075.jpg --mask_name=fish.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style " Enter
# tmux send-keys -t "2" "python style.py --input_name charlottesville-7-view.jpg --mask_name=frog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter
# tmux send-keys -t "2" "python style.py --input_name walen-log-view1.jpg --mask_name=goat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style" Enter


# NVIDIA inpainting

# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name 78308399-texture-of-red-gravel-red-rocky-floors-a-wall-of-red-gravel-stones-small-and-medium-sized-sharp-edge.jpg --mask_name=bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia"   Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name CF005_L_V_2012-c.jpg --mask_name=bird-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia " Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name=bird.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name 42482737980_94dcb940be_k.jpg --mask_name=butterfly.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000  --mode=nvidia" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name dan-woje-desertmix-surfacev2.jpg --mask_name=camel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name CP030_L_V_2013-c.jpg --mask_name=cat-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name walden-brush-view6.jpg --mask_name=cat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name savanna_1.jpg --mask_name=cheetah.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=nvidia" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name walden-brush2-view1.jpg --mask_name=cow.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name charlottesville-3-view15.jpg --mask_name=dog.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 14005790586_349d8028b5_k.jpg --mask_name=dolphin.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name charlottesville-7-view1.jpg --mask_name=deer.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 27186282509_126fb93851_k.jpg --mask_name=eagle.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia " Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea.jpg --mask_name=elephant.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=nvidia" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 3380728556_14574fff67_k.jpg --mask_name=fish-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 1388693307.jpg --mask_name=fish-3.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 65533488.jpg --mask_name=fish-4.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 247770075.jpg --mask_name=fish.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia " Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name charlottesville-7-view.jpg --mask_name=frog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name walen-log-view1.jpg --mask_name=goat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia" Enter