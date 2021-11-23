# admm compression in natural setting
# python -u main.py --config_file lenet_adv_config_warmup.yaml --stage pretrain
# python -u main.py --config_file lenet_adv_config_warmup.yaml --stage admm
# python -u main.py --config_file lenet_adv_config_warmup.yaml --stage retrain

# admm compression in adversarial setting
# python -u adv_main.py --config_file lenet_adv_config_warmup.yaml --stage pretrain &> logs/log_pretrain_8_83_all_layers.txt
python -u adv_main.py --config_file lenet_adv_config_warmup.yaml --stage admm  &> logs/log_admm_80_conv_fc1_adam.txt
python -u adv_main.py --config_file lenet_adv_config_warmup.yaml --stage retrain  &> logs/log_retrain_83_conv_fc1_adam.txt