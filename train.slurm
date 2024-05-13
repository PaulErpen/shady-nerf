#!/bin/sh

##the following lines starting with `#SBATCH` are directives to the scheduling system, slurm

## a job name
#SBATCH --job-name=paul-erpenstein-thesis

## the file the output your script procudes should be written to
## we recommend absolut file paths
## `~` is NOT expanded to your homedirectory here!
## either use a path relativ to the directory you use `sbatch` from to submit this job
## or specify an absolute path to make sure
#SBATCH --output=slurm_%j.log

## a self proclaimed time limit for your job, will be killed after time limit passes
## an absolute maximum will be applied though
## dont raise above absolute maximum because that might prevent your job from beeing scheduled
#SBATCH --time=01:00:0

## node to run on, remove one of the `#` in the beginning if you want to fixate it
## most likely its best to leave like this to get a node selected automatically
##SBATCH --nodelist=edna
##SBATCH --nodelist=skinner

## Please do not modify these. We cant stop you. It might work. It might not work.
## Though your jobs might just not get scheduled. You have been warned.
## (Nothing bad should happen. If you try and find something, we'd be happy if you notify us. Intentionall long term abuse of however is discouraged and can have consequences.)
#SBATCH --account=dlvc
#SBATCH --partition pDLVC
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem=8192mb
#SBATCH --gres=gpu:1

## Print some debug info
echo "================ ================ ================ ================"
echo "$(date)|$(hostname)|$(pwd)|$(id)"
echo "================ ================ ================ ================"
nvidia-smi
nvidia-smi -L
python3 --version
echo "================ ================ ================ ================"

## Installing my own module
python3 -m pip install --user -e ./shady-nerf/lodnelf-module

## Run your workload
srun --gres=gpu:1 python3 -m lodnelf.train.train_cmd --run_name "experiment_alpha_depth_2" --config "SimpleRedCarModel" --model_save_dir "models/experiment_alpha_depth_2" --data_dir "data"

## Make sure to wait for everything to complete
wait