'''configuration function:
'''
import json

import numpy as np
import yaml
import torch.nn as nn


class Config:
    def __init__(self, args):
        """
        read config file
        """
        config_dir = args.config_file
        stage = args.stage
        ### Own adaption for running as batch script
        self.stage = args.stage
        self.arch = args.arch

        if args.uniform:
            self.uniform = True
        else:
            self.uniform = False

        self.sparsity_type = args.sparsity_type
        self.pruning_rate = args.pruning_rate
        self.rate_from_config = args.rate_from_config
        self.run_id = args.run_id
        self.strategy_id = self.run_id
        self.strategy_id = f"{str(self.pruning_rate).replace('.', '')}_{self.strategy_id}"

        if self.pruning_rate == 0.1 and self.sparsity_type == "weight":
            self.strategy_id = self.strategy_id.replace("_", "0_")

        self.save_model = f"{self.arch}_{self.stage}_{self.sparsity_type}_{self.strategy_id}_{'HARP' if not self.uniform else ''}.pth.tar"

        if self.stage == 'retrain':
            self.load_model = f"{self.arch}_{'admm'}_{self.sparsity_type}_{self.strategy_id}_{'HARP' if not self.uniform else ''}.pth.tar"
        elif self.stage == "admm":
            if self.arch == "resnet18_adv":
                self.load_model = "resnet18_hydra_svhn.pth.tar"
            elif self.arch == "wrn_28_4":
                self.load_model = "wrn284_hydra_svhn.pth.tar"
            elif self.arch == "vgg16":
                self.load_model = "vgg16_hydra_svhn.pth.tar"
        else:
            self.load_model = None

        print(f"Stage is: {self.stage}")
        print(f"Arch is: {self.arch}")
        print(f"Uniform pruning: {self.uniform}")
        print(f"Sparsity type: {self.sparsity_type}")
        print(f"Pruning rate: {self.pruning_rate}")
        print(f"Run id: {self.run_id}")
        print(f"Pruning rate from config file: {self.rate_from_config}")
        print(f"Load model: {self.load_model}")
        print(f"Save model: {self.save_model}")

        strategies = json.load(open("strategies_harp_SVHN.json"))
        harp_stg = strategies[self.arch][self.sparsity_type][self.strategy_id]

        self.prune_stg = [[], [], []]  # [[conv_prates], [fc_prates], [shortcut-prates]]
        if self.arch == 'vgg16':
            assert len(harp_stg) == 16, '! ! {}-{}-{} is invalid'.format(self.arch, self.sparsity_type, self.run_id)
            self.prune_stg[0] = harp_stg[:13]
            self.prune_stg[1] = harp_stg[13:]
            self.prune_stg[2] = []
        elif self.arch == 'resnet18_adv':
            if self.sparsity_type == 'column':
                assert len(harp_stg) == 18, '! ! {}-{}-{} is invalid'.format(self.arch, self.sparsity_type, self.run_id)
                self.prune_stg[0] = harp_stg[:17]
                self.prune_stg[1] = harp_stg[17:]
                self.prune_stg[2] = list(np.take(harp_stg, (5, 9, 13)))
            if self.sparsity_type == 'weight':
                assert len(harp_stg) == 21, '! ! {}-{}-{} is invalid'.format(self.arch, self.sparsity_type, self.run_id)
                self.prune_stg[0] = harp_stg[:5] + harp_stg[6:10] + harp_stg[11:15] + harp_stg[16:20]
                self.prune_stg[1] = harp_stg[20:]
                self.prune_stg[2] = list(np.take(harp_stg, (5, 10, 15)))
        elif self.arch == 'wrn_28_4':
            if self.sparsity_type == 'column':
                assert len(harp_stg) == 26, '! ! {}-{}-{} is invalid'.format(self.arch, self.sparsity_type, self.run_id)
                self.prune_stg[0] = harp_stg[:25]
                self.prune_stg[1] = harp_stg[25:]
                self.prune_stg[2] = list(np.take(harp_stg, (1, 9, 17)))
            if self.sparsity_type == 'weight':
                assert len(harp_stg) == 29, '! ! {}-{}-{} is invalid'.format(self.arch, self.sparsity_type, self.run_id)
                self.prune_stg[0] = harp_stg[:1] + harp_stg[2:10] + harp_stg[11:19] + harp_stg[20:28]
                self.prune_stg[1] = harp_stg[28:]
                self.prune_stg[2] = list(np.take(harp_stg, (1, 10, 19)))
        else:
            raise NameError('Strategy check only supports vgg16, resnet18_adv, wrn_28_4 but no "{}"'.format(self.arch))

        if self.sparsity_type == "weight":
            self.sparsity_type = "irregular"

        try:
            with open(config_dir, "r") as stream:
                raw_dict = yaml.load(stream)

                # adv parameters
                self.epsilon = raw_dict['adv']['epsilon']
                self.num_steps = raw_dict['adv']['num_steps']
                self.step_size = raw_dict['adv']['step_size']
                self.random_start = raw_dict['adv']['random_start']
                self.loss_func = raw_dict['adv']['loss_func']
                self.width_multiplier = float(raw_dict['adv']['width_multiplier'])
                self.init_func = raw_dict['adv']['init_func']
                # general
                self.print_freq = raw_dict['general']['print_freq']
                self.resume = raw_dict['general']['resume']
                self.gpu = raw_dict['general']['gpu_id']
                # self.arch = raw_dict['general']['arch']
                self.workers = raw_dict['general']['workers']
                self.logging = raw_dict['general']['logging']
                self.log_dir = raw_dict['general']['log_dir']
                self.smooth_eps = raw_dict['general']['smooth_eps']
                self.alpha = raw_dict['general']['alpha']
                # self.sparsity_type = raw_dict['general']['sparsity_type']
                if stage != 'pretrain':
                    self._prune_ratios = raw_dict[self.arch]['prune_ratios']
                    if not self.rate_from_config:
                        if self.uniform:
                            for k, v in self._prune_ratios.copy().items():
                                self._prune_ratios[k] = 1 - self.pruning_rate
                        else:
                            self.insert_harp_strategy()

                self.name_encoder = {}
                self.lr = raw_dict[stage]['lr']
                self.lr_scheduler = raw_dict[stage]['lr_scheduler']
                self.optimizer = raw_dict[stage]['optimizer']
                # self.save_model = raw_dict[stage]['save_model']
                # self.load_model = None  # otherwise key error
                self.masked_progressive = None
                if stage != 'pretrain':
                    # self.load_model = raw_dict[stage]['load_model']
                    self.masked_progressive = raw_dict[stage]['masked_progressive']
                self.epochs = raw_dict[stage]['epochs']

                # if stage !='admm':
                #     self.warmup_epochs = raw_dict[stage]['warmup_epochs']
                #     self.warmup_lr = raw_dict[stage]['warmup_lr']
                self.warmup_epochs = raw_dict['pretrain']['warmup_epochs']
                self.warmup_lr = raw_dict['pretrain']['warmup_lr']

                self.admm = (stage == 'admm')
                self.masked_retrain = (stage == 'retrain')
                self.rho = None
                self.rhos = {}
                if stage == 'admm':
                    # admm_pruning
                    self.admm_epoch = raw_dict[stage]['admm_epoch']
                    self.rho = raw_dict[stage]['rho']
                    self.multi_rho = raw_dict[stage]['multi_rho']
                    self.verbose = raw_dict[stage]['verbose']
                # following variables assist the pruning algorithm

                self.masks = None
                self.zero_masks = None
                self.conv_names = []
                self.bn_names = []
                self.fc_names = []
                self.prune_ratios = {}
                self.model = None

        except yaml.YAMLError as exc:
            print(exc)

    def insert_harp_strategy(self):
        for i, (k, v) in enumerate(self._prune_ratios.copy().items()):
            if self.arch == "vgg16":
                assert len(self._prune_ratios) == 16

                if "conv" in k:
                    self._prune_ratios[k] = float(1 - self.prune_stg[0].pop(0))
                if "fc" in k:
                    self._prune_ratios[k] = float(1 - self.prune_stg[1].pop(0))

            elif self.arch == "resnet18_adv":
                assert len(self._prune_ratios) == 21

                if "fc" in k:
                    self._prune_ratios[k] = float(1 - self.prune_stg[1].pop(0))
                elif i in [7, 12, 17]:
                    self._prune_ratios[k] = float(1 - self.prune_stg[2].pop(0))
                else:
                    self._prune_ratios[k] = float(1 - self.prune_stg[0].pop(0))


            elif self.arch == "wrn_28_4":
                assert len(self._prune_ratios) == 29

                if "fc" in k:
                    self._prune_ratios[k] = float(1 - self.prune_stg[1].pop(0))
                elif i in [3, 12, 21]:
                    self._prune_ratios[k] = float(1 - self.prune_stg[2].pop(0))
                else:
                    self._prune_ratios[k] = float(1 - self.prune_stg[0].pop(0))

            else:
                print(f"Unknown arch {self.arch}")

        print(f"Replaced uniform strategy with HARP strategy")

    def prepare_pruning(self):
        if self.stage == 'pretrain':
            return
        self._extract_layer_names(self.model)
        for good_name, ratio in self._prune_ratios.items():
            self._encode(good_name)
        print("Using the following pruning ratios: ")
        for good_name, ratio in self._prune_ratios.items():
            self.prune_ratios[self.name_encoder[good_name]] = ratio
            print(f"{ratio}")
        for k in self.prune_ratios.keys():
            self.rhos[k] = self.rho  # this version we assume all rhos are equal

    def __str__(self):
        return str(self.__dict__)

    def _extract_layer_names(self, model):
        """
        Store layer name of different types in arrays for indexing
        """

        names = []
        for name, W in self.model.named_modules():
            names.append(name)
        print(names)
        for name, W in self.model.named_modules():
            name += '.weight'  # name in named_modules looks like module.features.0. We add .weight into it
            if isinstance(W, nn.Conv2d):
                self.conv_names.append(name)
            if isinstance(W, nn.BatchNorm2d):
                self.bn_names.append(name)
            if isinstance(W, nn.Linear):
                self.fc_names.append(name)

    def _encode(self, name):
        """
        Examples:
        conv1.weight -> conv           1                weight
                        conv1-> prefix   weight->postfix
                        conv->layer_type  1-> layer_id + 1  weight-> postfix
        Use buffer for efficient look up
        """
        prefix, postfix = name.split('.')
        dot_position = prefix.find('.')
        layer_id = ''
        for s in prefix:
            if s.isdigit():
                layer_id += s
        id_length = len(layer_id)
        layer_type = prefix[:-id_length]
        layer_id = int(layer_id) - 1
        if layer_type == 'conv' and len(self.conv_names) != 0:
            self.name_encoder[name] = self.conv_names[layer_id]
        elif layer_type == 'fc' and len(self.fc_names) != 0:
            self.name_encoder[name] = self.fc_names[layer_id]
        elif layer_type == 'bn' and len(self.bn_names) != 0:
            self.name_encoder[name] = self.bn_names[layer_id]
