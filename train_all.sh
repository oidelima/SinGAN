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

tmux send-keys -t "1" "echo sfjdskl"
tmux send-keys -t "1" "python3 main_train.py --input_name 181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08.jpg --mask_name bird-3.png --run_name=test --batch_size=1  --patch_size=96 --gpu=0 --alpha=10 --niter=2000 --niter=1 --mode train"   Enter
# tmux send-keys -t "2" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_1 --gpu=1 --alpha=0.1" Enter
# tmux send-keys -t "3" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_2 --gpu=2 --alpha=0.5" Enter
# tmux send-keys -t "4" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_3 --gpu=3 --alpha=1" Enter
# tmux send-keys -t "5" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_4 --gpu=4 --alpha=5" Enter
# tmux send-keys -t "6" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_5 --gpu=5 --alpha=10" Enter
# tmux send-keys -t "7" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=patch_6 --gpu=6 --patch_size=180" Enter
#tmux send-keys -t "8" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=patch_7 --gpu=7 --patch_size=250" Enter

#patch_sizes = [10, 25, 50, 75, 100, 150, 180]
#niter = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
#alpha = [0, 0.1, 0.5, 1, 5, 10, 20, 50]
