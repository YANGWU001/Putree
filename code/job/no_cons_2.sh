#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C V100
#SBATCH -t 12:00:00
#SBATCH --mem 16G
#SBATCH -p short
#SBATCH --job-name="0.0"
#SBATCH --output=result_0.0.txt



__conda_setup="$('/home/ywu19/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ywu19/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ywu19/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ywu19/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source activate pu_torch 
python ../no_cons_main_2.py

