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

# tmux send-keys -t "1" "python main_train.py --input_name 78308399-texture-of-red-gravel-red-rocky-floors-a-wall-of-red-gravel-stones-small-and-medium-sized-sharp-edge.jpg --mask_name=bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 25 52"   Enter
# tmux send-keys -t "1" "python main_train.py --input_name CF005_L_V_2012-c.jpg --mask_name=bird-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 41 68" Enter
# tmux send-keys -t "1" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name=bird.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 29 33" Enter
# tmux send-keys -t "1" "python main_train.py --input_name 42482737980_94dcb940be_k.jpg --mask_name=butterfly.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000  --mode=inpainting --fixed_eye_loc 19 58" Enter
# tmux send-keys -t "1" "python main_train.py --input_name dan-woje-desertmix-surfacev2.jpg --mask_name=camel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 28 55" Enter
# tmux send-keys -t "1" "python main_train.py --input_name CP030_L_V_2013-c.jpg --mask_name=cat-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 46 22" Enter
# tmux send-keys -t "1" "python main_train.py --input_name walden-brush-view6.jpg --mask_name=cat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 35 82" Enter
# tmux send-keys -t "1" "python main_train.py --input_name savanna_1.jpg --mask_name=cheetah.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=inpainting --fixed_eye_loc 37 45" Enter
# tmux send-keys -t "1" "python main_train.py --input_name walden-brush2-view1.jpg --mask_name=cow.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 30 12" Enter
# tmux send-keys -t "1" "python main_train.py --input_name charlottesville-3-view15.jpg --mask_name=dog.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 47 83" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 14005790586_349d8028b5_k.jpg --mask_name=dolphin.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 31 55" Enter
# tmux send-keys -t "2" "python main_train.py --input_name charlottesville-7-view1.jpg --mask_name=deer.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 45 61" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 27186282509_126fb93851_k.jpg --mask_name=eagle.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 38 81" Enter
# tmux send-keys -t "2" "python main_train.py --input_name ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea.jpg --mask_name=elephant.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=inpainting --fixed_eye_loc 47 40" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 3380728556_14574fff67_k.jpg --mask_name=fish-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 54 88" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 1388693307.jpg --mask_name=fish-3.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 79 56" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 65533488.jpg --mask_name=fish-4.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 28 43" Enter
# tmux send-keys -t "2" "python main_train.py --input_name 247770075.jpg --mask_name=fish.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 20 71" Enter
# tmux send-keys -t "2" "python main_train.py --input_name charlottesville-7-view.jpg --mask_name=frog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 37 59" Enter
# tmux send-keys -t "2" "python main_train.py --input_name walen-log-view1.jpg --mask_name=goat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=inpainting --fixed_eye_loc 52 73" Enter

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

tmux send-keys -t "1" "python random_crop.py --input_name swamp.jpg --mask_name=hippo.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 27 14" Enter
tmux send-keys -t "1" "python random_crop.py --input_name forest_floor_1.jpg --mask_name=iguana-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 29 73" Enter
tmux send-keys -t "1" " python random_crop.py --input_name 9184961-pink-pantip-flowers-on-the-floor.jpg --mask_name=iguana.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 16 38" Enter
tmux send-keys -t "1" "python random_crop.py --input_name branches.jpg --mask_name=koala.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 30 47" Enter
tmux send-keys -t "1" "python random_crop.py --input_name woods_orange.jpg --mask_name=lion-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 49 51" Enter
tmux send-keys -t "1" "python random_crop.py --input_name woods_yellow.jpg --mask_name=lion.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 48 49" Enter
tmux send-keys -t "1" "python random_crop.py --input_name CF026_R_V_2012-c.jpg --mask_name=llama.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 25 79" Enter
tmux send-keys -t "1" "python random_crop.py --input_name walden-brush-view8.jpg --mask_name=monkey.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 46 49 " Enter
tmux send-keys -t "1" "python random_crop.py --input_name woods_3.jpg --mask_name=panda.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 50 35" Enter
tmux send-keys -t "1" "python random_crop.py --input_name icy.jpg --mask_name=penguin.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 42 57" Enter
tmux send-keys -t "1" "python random_crop.py --input_name snowy.jpg --mask_name=polar_bear.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 43 73" Enter
tmux send-keys -t "1" "python random_crop.py --input_name CP035_L_V_2013-c.jpg --mask_name=prairie_dog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 56 18" Enter
tmux send-keys -t "1" "python random_crop.py --input_name charlottesville-3-1.jpg --mask_name=rabbit-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 56 29" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name MAT18-4.jpg --mask_name=rabbit-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 47 59" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name walden-tree3-view1.jpg --mask_name=rabbit.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 41 62" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name CP029_L_V_2012-c.jpg --mask_name=racoon.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 38 40" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name water_3.jpg --mask_name=ray.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 32 25" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name forest_1.jpg --mask_name=rhino.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 42 66" Enter
tmux send-keys -t "1" "python random_crop.py --input_name CP002_R_V_2013-c.jpg --mask_name=rooster.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 24 55" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name trees.jpg --mask_name=seagull.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 38 25" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name water.jpg --mask_name=shark-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 32 55" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name water_2.jpg --mask_name=shark.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 30 53" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name savanna_2.jpg --mask_name=sheep.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 23 59" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name forest-floor-wallpaper.jpg --mask_name=squirrel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 14 22" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name fall.jpg --mask_name=tiger-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 18 28" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name forest_2.jpg --mask_name=tiger.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 22 40" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name bilde_t_1.jpg --mask_name=turtle.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 34 32" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name 29736858514_b497dd956c_k.jpg --mask_name=wolf-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 20 27" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name snowy_2.jpg --mask_name=wolf.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 43 46" Enter 
tmux send-keys -t "1" "python random_crop.py --input_name grass.jpg --mask_name=zebra.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train --fixed_eye_loc 22 45" Enter 


# Style transfer

# tmux send-keys -t "1" "python style.py --input_name 78308399-texture-of-red-gravel-red-rocky-floors-a-wall-of-red-gravel-stones-small-and-medium-sized-sharp-edge.jpg --mask_name=bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 25 52"   Enter
# tmux send-keys -t "1" "python style.py --input_name CF005_L_V_2012-c.jpg --mask_name=bird-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 41 68" Enter
# tmux send-keys -t "1" "python style.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name=bird.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 29 33" Enter
# tmux send-keys -t "1" "python style.py --input_name 42482737980_94dcb940be_k.jpg --mask_name=butterfly.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000  --mode=style --fixed_eye_loc 19 58" Enter
# tmux send-keys -t "1" "python style.py --input_name dan-woje-desertmix-surfacev2.jpg --mask_name=camel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 28 55" Enter
# tmux send-keys -t "1" "python style.py --input_name CP030_L_V_2013-c.jpg --mask_name=cat-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 46 22" Enter
# tmux send-keys -t "1" "python style.py --input_name walden-brush-view6.jpg --mask_name=cat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 35 82" Enter
# tmux send-keys -t "1" "python style.py --input_name savanna_1.jpg --mask_name=cheetah.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=style --fixed_eye_loc 37 45" Enter
# tmux send-keys -t "1" "python style.py --input_name walden-brush2-view1.jpg --mask_name=cow.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 30 12" Enter
# tmux send-keys -t "1" "python style.py --input_name charlottesville-3-view15.jpg --mask_name=dog.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 47 83" Enter
# tmux send-keys -t "2" "python style.py --input_name 14005790586_349d8028b5_k.jpg --mask_name=dolphin.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 31 55" Enter
# tmux send-keys -t "2" "python style.py --input_name charlottesville-7-view1.jpg --mask_name=deer.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 45 61" Enter
# tmux send-keys -t "2" "python style.py --input_name 27186282509_126fb93851_k.jpg --mask_name=eagle.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 38 81" Enter
# tmux send-keys -t "2" "python style.py --input_name ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea.jpg --mask_name=elephant.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=style --fixed_eye_loc 47 40" Enter
# tmux send-keys -t "2" "python style.py --input_name 3380728556_14574fff67_k.jpg --mask_name=fish-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 54 88" Enter
# tmux send-keys -t "2" "python style.py --input_name 1388693307.jpg --mask_name=fish-3.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 79 56" Enter
# tmux send-keys -t "2" "python style.py --input_name 65533488.jpg --mask_name=fish-4.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 28 43" Enter
# tmux send-keys -t "2" "python style.py --input_name 247770075.jpg --mask_name=fish.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 20 71" Enter
# tmux send-keys -t "2" "python style.py --input_name charlottesville-7-view.jpg --mask_name=frog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 37 59" Enter
# tmux send-keys -t "2" "python style.py --input_name walen-log-view1.jpg --mask_name=goat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=style --fixed_eye_loc 52 73" Enter


# NVIDIA inpainting

# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name 78308399-texture-of-red-gravel-red-rocky-floors-a-wall-of-red-gravel-stones-small-and-medium-sized-sharp-edge.jpg --mask_name=bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 25 52"   Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name CF005_L_V_2012-c.jpg --mask_name=bird-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 41 68" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name=bird.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 29 33" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name 42482737980_94dcb940be_k.jpg --mask_name=butterfly.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000  --mode=nvidia --fixed_eye_loc 19 58" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name dan-woje-desertmix-surfacev2.jpg --mask_name=camel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 28 55" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name CP030_L_V_2013-c.jpg --mask_name=cat-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 46 22" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name walden-brush-view6.jpg --mask_name=cat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 35 82" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name savanna_1.jpg --mask_name=cheetah.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=nvidia --fixed_eye_loc 37 45" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name walden-brush2-view1.jpg --mask_name=cow.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 30 12" Enter
# tmux send-keys -t "1" "python nvidia_inpainting.py --input_name charlottesville-3-view15.jpg --mask_name=dog.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 47 83" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 14005790586_349d8028b5_k.jpg --mask_name=dolphin.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 31 55" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name charlottesville-7-view1.jpg --mask_name=deer.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 45 61" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 27186282509_126fb93851_k.jpg --mask_name=eagle.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 38 81" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea.jpg --mask_name=elephant.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=nvidia --fixed_eye_loc 47 40" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 3380728556_14574fff67_k.jpg --mask_name=fish-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 54 88" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 1388693307.jpg --mask_name=fish-3.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 79 56" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 65533488.jpg --mask_name=fish-4.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 28 43" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name 247770075.jpg --mask_name=fish.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 20 71" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name charlottesville-7-view.jpg --mask_name=frog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 37 59" Enter
# tmux send-keys -t "2" "python nvidia_inpainting.py --input_name walen-log-view1.jpg --mask_name=goat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=nvidia --fixed_eye_loc 52 73" Enter