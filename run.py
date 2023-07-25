"""Load model, data and corresponding configs. Trigger training."""
import os
import pathlib
import sys
#import idr_torch

import numpy as np
import torch
import torch.distributed as dist

from npt.column_encoding_dataset import (ColumnEncodingDataset,
                                         NPTDataset)
from npt.configs import build_parser
from npt.train import Trainer
from npt.utils.model_init_utils import (init_model_opt_scaler_from_dataset, 
                                        setup_ddp_model)

# Environment variables set by torch.distributed.launch
if torch.cuda.is_available():
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

def main(args):
    """Load model, data, configs, start training."""
    args = setup_args(args)
    run_cv(args=args)


def setup_args(args):
    print('Configuring arguments...')


    if args.np_seed == -1:
        args.np_seed = np.random.randint(0, 1000)
    if args.torch_seed == -1:
        args.torch_seed = np.random.randint(0, 1000)

    if not isinstance(args.model_augmentation_bert_mask_prob, dict):
        print('Reading dict for model_augmentation_bert_mask_prob.')
        exec(
            f'args.model_augmentation_bert_mask_prob = '
            f'{args.model_augmentation_bert_mask_prob}')

    if not args.model_bert_augmentation:
        for value in args.model_augmentation_bert_mask_prob.values():
            assert value == 0
        for value in args.model_label_bert_mask_prob.values():
            assert value == 1

    # Set seeds
    np.random.seed(args.np_seed)

    # Resolve CUDA device(s)
    if args.exp_use_cuda and torch.cuda.is_available():
        if args.exp_device is not None:
            print(f'Running model with CUDA on device {args.exp_device}.')
            exp_device = args.exp_device
        else:
            print(f'Running model with CUDA')
            exp_device = 'cuda:0'
    else:
        print('Running model on CPU.')
        exp_device = 'cpu'

    args.exp_device = exp_device

    return args


def run_cv(args):
    
    c = args

    run_cv_splits(args, c)

def run_cv_splits(args, c):
    
    if c.mp_distributed:
        dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=WORLD_SIZE,
                rank=WORLD_RANK)


    for run in range(args.n_runs):

        c.torch_seed = args.original_torch_seed + run
        print(f'Setting torch seed to {c.torch_seed}')
        torch.manual_seed(c.torch_seed)
        dataset = ColumnEncodingDataset(c)
        

        #######################################################################
        # Distributed Setting
        if c.mp_distributed:
            
            dataset.load_next_cv_split()
            dataset.dataset_gen = None
            
            torch.cuda.set_device(LOCAL_RANK)
            model, optimizer, scaler = init_model_opt_scaler_from_dataset(
            dataset=dataset, c=c, device=LOCAL_RANK)
            
            dist.barrier()
            
            model = setup_ddp_model(model=model, c=c, device=LOCAL_RANK)
                    
            distributed_dataset = NPTDataset(dataset)
            dist_args = {
            'world_size': WORLD_SIZE,
            'rank': WORLD_RANK,
            'gpu': LOCAL_RANK}
            
            trainer = Trainer(
                model=model, optimizer=optimizer, scaler=scaler, c=c,
                cv_index=0,
                dataset=dataset,
                torch_dataset=distributed_dataset, distributed_args=dist_args, 
                ad=c.ad)
            trainer.train_and_eval()
            dist.barrier()

        else:

            print(f'CV Index: {run}')

            print(f'Train-test Split {run}/{c.n_runs}')

            if c.exp_n_runs < dataset.n_cv_splits:
                print(
                    f'c.exp_n_runs = {c.exp_n_runs}. '
                    f'Stopping at {c.exp_n_runs} splits.')

            # New wandb logger for each run
            if run > 0:
                args.cv_index = run

            #######################################################################
            # Load New CV Split
            dataset.load_next_cv_split()

            #######################################################################
            # Initialise Model
            model, optimizer, scaler = init_model_opt_scaler_from_dataset(
                dataset=dataset, c=c, device=c.exp_device)

            #######################################################################
            # Run training
            trainer = Trainer(
                model=model, optimizer=optimizer, scaler=scaler,
                c=c, cv_index=run,
                dataset=dataset, ad=args.ad)
            trainer.train_and_eval()


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    main(args)
