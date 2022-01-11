#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
ARCH="wrn_28_4" # vgg16, wrn_28_4, resnet18_adv
SPARSITY_TYPE="weight"  # weight, column
RATE="0.1"
ID="1"
UNIFORM="false"

if [[ $UNIFORM == "true" ]];
then
  echo "Uniform pruning"
  # python -u adv_main.py --config_file config.yaml --stage admm --uniform --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee  logs/${ARCH}_admm_${SPARSITY_TYPE}_${RATE}_cifar10_SGD.log
  python -u adv_main.py --config_file config.yaml --stage retrain --uniform --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee logs/${ARCH}_retrain_${SPARSITY_TYPE}_${RATE}_cifar10_SGD.log
else
  echo "HARP mask pruning"
  # python -u adv_main.py --config_file config.yaml --stage admm --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee logs/${ARCH}_admm_${SPARSITY_TYPE}_${RATE}_${ID}_cifar10_SGD.log
  python -u adv_main.py --config_file config.yaml --stage retrain --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee logs/${ARCH}_retrain_${SPARSITY_TYPE}_${RATE}_${ID}_cifar10_SGD.log
fi

# Check if gpu available: sinfo_t_idle
# check all jobs: squeue
# check specific job: scontrol show job <jobid> | grep -i State
# cancel job: scancel <jobid>