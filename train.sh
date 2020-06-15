tmux new -d -s 0
tmux new -d -s 1
tmux new -d -s 2
tmux new -d -s 3
tmux new -d -s 4
tmux new -d -s 5
tmux new -d -s 6
tmux new -d -s 7
tmux new -d -s 8
tmux new -d -s 9

tmux send-keys -t "0" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_0 --gpu=0 --alpha=0" Enter
tmux send-keys -t "1" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_1 --gpu=1 --alpha=0.1" Enter
tmux send-keys -t "2" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_2 --gpu=2 --alpha=0.5" Enter
tmux send-keys -t "3" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_3 --gpu=3 --alpha=1" Enter
tmux send-keys -t "4" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_4 --gpu=4 --alpha=5" Enter
tmux send-keys -t "5" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_5 --gpu=5 --alpha=10" Enter
tmux send-keys -t "6" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_6 --gpu=6 --alpha=20" Enter
tmux send-keys -t "7" "python main_train.py --input_name 94547580-ivy-growing-on-the-forest-floor.jpg --mask_name beetle-12.gif --run_name=alpha_7 --gpu=7 --alpha=50" Enter
