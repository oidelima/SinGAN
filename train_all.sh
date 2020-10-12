# tmux new -d -s 0
tmux new -d -s 1
# tmux new -d -s 2
# tmux new -d -s 3
# tmux new -d -s 4
# tmux new -d -s 5
#tmux new -d -s 6
# tmux new -d -s 7
# tmux new -d -s 8
# tmux new -d -s 9


tmux send-keys -t "1" "python main_train.py --input_name 78308399-texture-of-red-gravel-red-rocky-floors-a-wall-of-red-gravel-stones-small-and-medium-sized-sharp-edge.jpg --mask_name=bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train"   Enter
tmux send-keys -t "1" "python main_train.py --input_name CF005_L_V_2012-c.jpg --mask_name=bird-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train " Enter
tmux send-keys -t "1" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name=bird.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name 42482737980_94dcb940be_k.jpg --mask_name=butterfly.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000  --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name dan-woje-desertmix-surfacev2.jpg --mask_name=camel.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name CP030_L_V_2013-c.jpg --mask_name=cat-2.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name 2422553984_fc557f797a_b.jpg --mask_name=cat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name forest-floor.jpg --mask_name=cheetah.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name walden-brush2-view1.jpg --mask_name=cow.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name charlottesville-3-view15.jpg --mask_name=dog.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name 14005790586_349d8028b5_k.jpg --mask_name=dolphin.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name charlottesville-7-view1.jpg --mask_name=deer.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name 27186282509_126fb93851_k.jpg --mask_name=eagle.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train " Enter
tmux send-keys -t "1" "python main_train.py --input_name ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea.jpg --mask_name=elephant.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000  --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name 2081085051.jpg --mask_name=fish-2.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name 1388693307.jpg --mask_name=fish-3.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name 65533488.jpg --mask_name=fish-4.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name 247770075.jpg --mask_name=fish.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train " Enter
tmux send-keys -t "1" "python main_train.py --input_name charlottesville-7-view.jpg --mask_name=frog.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter
tmux send-keys -t "1" "python main_train.py --input_name walen-log-view1.jpg --mask_name=goat.png --run_name=test --batch_size=1  --patch_size=128 --gpu=0 --alpha=10 --niter=2000 --mode=train" Enter