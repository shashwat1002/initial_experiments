#!/bin/bash
#!/bin/bash
#SBATCH -n 16
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=20
#SBATCH --gres=gpu:2
#SBATCH --mail-user=shashwat.s@research.iiit.ac.in
#SBATCH --mail-type=ALL
module load cuda/10.0
module load cudnn/7.6-cuda-10.0

eval "$(conda shell.bash hook)"
conda activate nlp_coursework



python train.py --model_name=roberta_model --batch_size=800 --learning_rate=1e-3 --num_epochs=200
