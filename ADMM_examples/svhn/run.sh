#admm compression in natural setting
#python -u main.py --config_file config.yaml --stage pretrain
#python -u main.py --config_file config.yaml --stage admm
#python -u main.py --config_file config.yaml --stage retrain

#admm compression in adversarial setting
# python -u adv_main.py --config_file config.yaml --stage pretrain &> logs/vgg16_pretrain_no_warmup.log
python -u adv_main.py --config_file config.yaml --stage admm # &> logs/wrn_28_4_admm_tes.log
# python -u adv_main.py --config_file config.yaml --stage retrain &> logs/resnet18_4_to_2_retrain_all_layers_column_200_epochs.log