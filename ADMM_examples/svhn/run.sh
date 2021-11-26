#admm compression in natural setting
#python -u main.py --config_file config.yaml --stage pretrain
#python -u main.py --config_file config.yaml --stage admm
#python -u main.py --config_file config.yaml --stage retrain

#admm compression in adversarial setting
# python -u adv_main.py --config_file config.yaml --stage pretrain 2>&1 | tee logs/vgg16_pretrain_new_adam.log
python -u adv_main.py --config_file config.yaml --stage admm 2>&1 | tee -a logs/resnet18_admm_weight_10_svhn_HARP_fgsm_warmup.log
python -u adv_main.py --config_file config.yaml --stage retrain 2>&1 | tee -a logs/resnet18_retrain_weight_1_svhn_HARP.log