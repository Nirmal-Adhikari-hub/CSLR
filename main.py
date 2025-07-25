import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
import shutil
from distutils.dir_util import copy_tree
import inspect
import time
from collections import OrderedDict

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation
from torch.cuda.amp import autocast as autocast
import datetime

import os
import shutil

import torch.distributed as dist


def prepare_work_dir(work_dir):
    if os.path.exists(work_dir):
        try:
            # Check if directory is empty
            if len(os.listdir(work_dir)) == 0:
                print(f"🧹 Cleaning empty work_dir: {work_dir}")
                shutil.rmtree(work_dir, ignore_errors=True)
            else:
                print(f"⚠️  Warning: work_dir '{work_dir}' already exists and is not empty.")
                print(f"ℹ️  Training will resume or overwrite depending on config/checkpoint logic.")
        except Exception as e:
            print(f"⚠️  Could not clean work_dir safely: {e}")
    os.makedirs(work_dir, exist_ok=True)


class Processor():
    def __init__(self, arg):
        # Distributed setup
        self.arg = arg
        
        # Ensure work_dir ends with '/'
        if not self.arg.work_dir.endswith('/'):
            self.arg.work_dir = os.path.join(self.arg.work_dir, '')

        # Only rank 0 should perform file copy operations to avoid race conditions
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Copy this script
            try:
                shutil.copy2(__file__, self.arg.work_dir)
            except FileNotFoundError:
                print(f"⚠️  Could not copy current file (__file__)")

            # Copy baseline config
            try:
                shutil.copy2('./configs/baseline.yaml', self.arg.work_dir)
            except FileNotFoundError:
                print(f"⚠️  baseline.yaml not found.")

            # Ignore cache files
            ignore_patterns = shutil.ignore_patterns('__pycache__', '*.pyc')

            # Copy slowfast_modules
            slowfast_dst = os.path.join(self.arg.work_dir, 'slowfast_modules')
            shutil.copytree(
                'slowfast_modules', slowfast_dst,
                dirs_exist_ok=True, ignore=ignore_patterns
            )

            # Copy modules
            modules_dst = os.path.join(self.arg.work_dir, 'modules')
            shutil.copytree(
                'modules', modules_dst,
                dirs_exist_ok=True, ignore=ignore_patterns
            )

        # Barrier: wait for rank 0 to finish copying
        if dist.is_initialized():
            dist.barrier()

        # Recorder setup
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        
        # Decide whether to load slowfast pkl
        if self.arg.load_checkpoints or self.arg.load_weights:
            self.load_slowfast_pkl = False
        else:
            self.load_slowfast_pkl = True

        # Save args and fix randomness
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)

        # Device setup for DDP
        self.device = torch.device(f"cuda:{self.arg.local_rank}")
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)

        # Data structures
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.arg.optimizer_args['num_epoch'] = self.arg.num_epoch

        # Flatten slowfast_args to list
        slowfast_args = []
        for key, value in self.arg.slowfast_args.items():
            slowfast_args.extend([key, value])
        self.arg.slowfast_args = slowfast_args

        # Load model and optimizer
        self.model, self.optimizer = self.loading()

    def start(self):
        if self.arg.phase == 'train':
            best_dev = 100.0
            best_epoch = self.arg.optimizer_args['start_epoch'] - 1
            total_time = 0
            epoch_time = 0
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                epoch_time = time.time()
                # train end2end model
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder)
                if eval_model:
                    dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                       'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                    self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                    test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                        "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                    self.recoder.print_log("Test WER: {:05.2f}%".format(test_wer))
                if dev_wer < best_dev:
                    best_dev = dev_wer
                    best_epoch = epoch
                    model_path = "{}_best_model.pt".format(self.arg.work_dir)
                    self.save_model(epoch, model_path)
                    self.recoder.print_log('Save best model')
                self.recoder.print_log('Best_dev: {:05.2f}, Epoch : {}'.format(best_dev, best_epoch))
                if save_model:
                    model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer, epoch)
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)
                epoch_time = time.time() - epoch_time
                total_time += epoch_time
                torch.cuda.empty_cache()
                self.recoder.print_log('Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time)//60, int(epoch_time)%60))
            self.recoder.print_log('Training costs {} hours {} mins {} seconds'.format(int(total_time)//60//60, int(total_time)//60%60, int(total_time)%60))
            torch.cuda.empty_cache()
        elif self.arg.phase == 'test':
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                print('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            # train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
            #                      "train", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
                               "dev", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def loading(self):
        # self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
            load_pkl=self.load_slowfast_pkl,
            slowfast_config=self.arg.slowfast_config,
            slowfast_args = self.arg.slowfast_args
        )
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        # self.kernel_sizes = model.conv1d.kernel_size

        # self.kernel_sizes = model.module.conv1d.kernel_size
        if hasattr(model, "module") and hasattr(model.module, "conv1d"):
            self.kernel_sizes = model.module.conv1d.kernel_size
        else:
            raise AttributeError("Model missing conv1d layer. Check architecture.")

        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        '''Original code'''
        # model = model.to(self.device.output_device)
        # if len(self.device.gpu_list) > 1:
        #     raise ValueError("AMP equipped with DataParallel has to manually write autocast() for each forward function, you can choose to do this by yourself")
        #     # model.conv2d = nn.DataParallel(model.conv2d, device_ids=self.device.gpu_list, output_device=self.device.output_device)
        #     # model = convert_model(model)
        # model.cuda()

        '''Modified code'''
        model = model.to(self.device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.arg.local_rank], output_device=self.arg.local_rank
        )
        return model

    def load_model_weights(self, model, weight_path):
        # map all tensors to this rank’s GPU
        map_loc = lambda storage, loc: storage.cuda(self.arg.local_rank)
        state_dict = torch.load(weight_path, map_location=map_loc)
        # state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        # Strip a leading "module." from every key if present
        new_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            new_dict[new_key] = v
        # if you had any additional modifications, handle them here...
        return new_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        # remap checkpoint tensors onto this rank’s GPU
        map_loc = lambda storage, loc: storage.cuda(self.arg.local_rank)
        state_dict = torch.load(self.arg.load_checkpoints, map_location=map_loc)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            # move optimizer state to the correct device
            optimizer.to(self.device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(self.feeder), self.arg.work_dir)
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False]) 
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, kernel_size= self.kernel_sizes, dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")
    def init_fn(self, worker_id):
        # np.random.seed(int(self.arg.random_seed)+int(self.arg.device)+worker_id)
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    def build_dataloader(self, dataset, mode, train_flag):
        ''' Added for the distributed training '''
        if train_flag:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True
            )
        else:
            sampler = None
        '''END of code addition'''    
        
        ''' Original code'''
        # return torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
        #     shuffle=train_flag,
        #     drop_last=train_flag,
        #     num_workers=self.arg.num_worker,  # if train_flag else 0
        #     collate_fn=self.feeder.collate_fn,
        #     pin_memory=True,
        #     worker_init_fn=self.init_fn,
        # )

        '''Modified code'''
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=(sampler is None),
            drop_last=train_flag,
            sampler=sampler,
            num_workers=self.arg.num_worker,
            collate_fn=self.feeder.collate_fn,
            pin_memory=True,
            worker_init_fn=self.init_fn,
        )


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# if __name__ == '__main__':
#     sparser = utils.get_parser()
#     p = sparser.parse_args()
#     # p.config = "baseline_iter.yaml"
#     if p.config is not None:
#         with open(p.config, 'r') as f:
#             try:
#                 default_arg = yaml.load(f, Loader=yaml.FullLoader)
#             except AttributeError:
#                 default_arg = yaml.load(f)
#         key = vars(p).keys()
#         for k in default_arg.keys():
#             if k not in key:
#                 print('WRONG ARG: {}'.format(k))
#                 assert (k in key)
#         sparser.set_defaults(**default_arg)
#     args = sparser.parse_args()
#     with open(f"./configs/{args.dataset}.yaml", 'r') as f:
#         args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

#     args.local_rank = setup_ddp()
#     processor = Processor(args)
#     #utils.pack_code("./", args.work_dir)
#     processor.start()


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)

    args = sparser.parse_args()

    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    args.local_rank = setup_ddp()

    # ✅ Insert safe work_dir setup here
    prepare_work_dir(args.work_dir)

    processor = Processor(args)
    # utils.pack_code("./", args.work_dir)
    processor.start()
