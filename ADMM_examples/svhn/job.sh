#!/bin/bash
ARCH="resnet18_adv" # vgg16, wrn_28_4, resnet18_adv
SPARSITY_TYPE="weight"  # weight, column
RATE="0.01"
ID="0"
UNIFORM="true"

if [[ $UNIFORM == "true" ]];
then
  echo "Uniform pruning"
  python -u adv_main.py --config_file config.yaml --stage admm --uniform --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee  logs/${ARCH}_admm_${SPARSITY_TYPE}_${RATE}_svhn.log
  python -u adv_main.py --config_file config.yaml --stage retrain --uniform --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee logs/${ARCH}_retrain_${SPARSITY_TYPE}_${RATE}_svhn.log
else
  echo "HARP mask pruning"
  python -u adv_main.py --config_file config.yaml --stage admm --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee logs/${ARCH}_admm_${SPARSITY_TYPE}_${RATE}_${ID}_svhn_HARP.log
  python -u adv_main.py --config_file config.yaml --stage retrain --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee logs/${ARCH}_retrain_${SPARSITY_TYPE}_${RATE}_${ID}_svhn_HARP.log
fi