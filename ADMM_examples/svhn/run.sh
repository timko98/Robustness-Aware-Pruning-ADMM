#!/bin/bash
#admm compression in natural setting
#python -u main.py --config_file config.yaml --stage pretrain
#python -u main.py --config_file config.yaml --stage admm
#python -u main.py --config_file config.yaml --stage retrain

#admm compression in adversarial setting
# python -u adv_main.py --config_file config.yaml --stage pretrain 2>&1 | tee logs/vgg16_pretrain_new_adam.log
# python -u adv_main.py --config_file config.yaml --stage admm 2>&1 | tee -a logs/resnet18_admm_weight_10_svhn_HARP_fgsm_warmup.log
# python -u adv_main.py --config_file config.yaml --stage retrain 2>&1 | tee -a logs/resnet18_retrain_weight_1_svhn_HARP.log

ARCH="vgg16"
SPARSITY_TYPE="weight"
RATE="0.1"
ID="0"
UNIFORM="true"

if [[ $UNIFORM == "true" ]];
then
  echo "Uniform pruning"
  python -u adv_main.py --config_file config.yaml --stage admm --uniform --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee  logs/${ARCH}_admm_${SPARSITY_TYPE}_${RATE}_svhn_fgsm_warmup.log
  python -u adv_main.py --config_file config.yaml --stage retrain --uniform --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee logs/${ARCH}_retrain_${SPARSITY_TYPE}_${RATE}_svhn_fgsm_warmup.log
else
  echo "HARP mask pruning"
  python -u adv_main.py --config_file config.yaml --stage admm --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee logs/${ARCH}_admm_${SPARSITY_TYPE}_${RATE}_svhn_HARP_fgsm_warmup.log
  python -u adv_main.py --config_file config.yaml --stage retrain --arch $ARCH --sparsity_type $SPARSITY_TYPE --pruning_rate $RATE --run_id $ID 2>&1 | tee logs/${ARCH}_retrain_${SPARSITY_TYPE}_${RATE}_svhn_HARP_fgsm_warmup.log
fi