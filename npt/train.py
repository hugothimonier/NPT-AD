"""Contains main training operations."""

import gc, os, pickle, time, sys, glob
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from torchmetrics import (AveragePrecision, AUROC)
from torchmetrics.classification import BinaryPrecisionRecallCurve
from sklearn.metrics import precision_recall_fscore_support as prf

from npt.column_encoding_dataset import ColumnEncodingDataset, NPTDataset
from npt.loss import Loss
from npt.optim import LRScheduler
from npt.utils.batch_utils import collate_with_pre_batching
from npt.utils.encode_utils import torch_cast_to_dtype
from npt.utils.eval_checkpoint_utils import EarlyStopCounter

from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
            self, model, optimizer, scaler, c, cv_index,
            dataset: ColumnEncodingDataset = None,
            torch_dataset: NPTDataset = None,
            distributed_args=None,
            ad:bool=False,):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = LRScheduler(
            c=c, name=c.exp_scheduler, optimizer=optimizer)
        self.c = c
        self.is_distributed = False
        self.rank = None
        self.dataset = dataset
        self.torch_dataset = torch_dataset
        self.max_epochs = self.get_max_epochs()
        self.ad = ad
        self.n_iter = 0

        # Data Loading
        self.data_loader_nprocs = (
            cpu_count() if c.data_loader_nprocs == -1
            else c.data_loader_nprocs)

        if self.data_loader_nprocs > 0:
            print(
                f'Distributed data loading with {self.data_loader_nprocs} '
                f'processes.')

        # Only needs to be set in distributed setting; otherwise, submodules
        # such as Loss and EarlyStopCounter use c.exp_device for tensor ops.
        self.gpu = None

        if distributed_args is not None:
            print('Loaded in DistributedDataset.')
            self.is_distributed = True
            self.world_size = distributed_args['world_size']
            self.rank = distributed_args['rank']
            self.gpu = distributed_args['gpu']
            
        self.print = ((self.is_distributed and self.rank==0) or 
                      (not self.is_distributed))
        
        if (self.is_distributed and self.rank==0) or (not self.is_distributed):
            ## for logging only
            str_train = str(self.c.model_augmentation_bert_mask_prob['train'])
            str_val = str(self.c.model_augmentation_bert_mask_prob['val'])
            str_type_embed = str(self.c.model_feature_type_embedding)[0]
            str_index_embed = str(self.c.model_feature_index_embedding)[0]
            job_name = ('TORCH_SEED_{torch_seed}__'
                        '{dataset}__job_{jobid}__bs_{batchsize}__lr_{lr}'
                        '__nsteps_{nsteps}__hdim_{hdim}__'
                        'trainmaskprob_{trainmaskprob}__'
                        'valmaskprob_{valmaskprob}__'
                        'num_hds_{num_heads}__'
                        'stcking_depth_{model_stacking_depth}__'
                        'type_embed_{type_embbedding}__'
                        'index_embed_{index_embedding}__'
                        'ssl_{ssl}')
            job_name= job_name.format(
                      np_seed=self.c.np_seed,
                      torch_seed=self.c.torch_seed,
                      dataset=c.data_set.upper(),
                      jobid=os.environ.get('SLURM_JOBID','') ,
                      batchsize=str(self.c.exp_batch_size),
                      lr=self.c.exp_lr,
                      nsteps=int(self.c.exp_num_total_steps),
                      hdim=self.c.model_dim_hidden,
                      trainmaskprob=str_train,
                      valmaskprob=str_val,
                      num_heads=self.c.model_num_heads,
                      model_stacking_depth=self.c.model_stacking_depth,
                      type_embbedding=str_type_embed,
                      index_embedding=str_index_embed,
                      ssl=str(self.c.model_is_semi_supervised)[0],
                      )
            if self.c.model_is_semi_supervised:
                job_name += f'__sslepochs_{self.c.exp_ssl_epochs}'
                                                                 
                                                            
            loss_dir = os.path.join('./tblogs', c.data_set, job_name)
            if not os.path.isdir(loss_dir):
                os.makedirs(loss_dir)
            self.writer = SummaryWriter(loss_dir)


        if c.exp_checkpoint_setting is None and c.exp_eval_test_at_end_only:
            raise Exception(
                'User is not checkpointing, but aims to evaluate the best '
                'performing model at the end of training. Please set '
                'exp_checkpoint_setting to "best_model" to do so.')

        self.early_stop_counter = EarlyStopCounter(
            c=c, data_cache_prefix=dataset.model_cache_path,
            metadata=dataset.metadata, cv_index=cv_index,
            n_splits=min(dataset.n_cv_splits, c.exp_n_runs),
            device=self.gpu)

        # Initialize from checkpoint, if available
        num_steps = 0

        if self.c.exp_load_from_checkpoint:
            checkpoint = self.early_stop_counter.get_most_recent_checkpoint()
            if checkpoint is not None:
                del self.model
                gc.collect()
                checkpoint_epoch, (
                    self.model, self.optimizer, self.scaler,
                    num_steps) = checkpoint

        self.loss = Loss(
            self.c, dataset.metadata,
            device=self.gpu, tradeoff_annealer=None,
            is_minibatch_sgd=self.c.exp_minibatch_sgd)

        if self.c.exp_eval_every_epoch_or_steps == 'steps':
            self.last_eval = 0

    def get_distributed_dataloader(self, epoch):
        if not self.is_distributed:
            raise Exception

        sampler = torch.utils.data.distributed.DistributedSampler(
            self.torch_dataset,
            num_replicas=self.world_size,
            rank=self.rank)

        dataloader = torch.utils.data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=1,  # The dataset is already batched.
            shuffle=False,
            num_workers=self.data_loader_nprocs,
            pin_memory=False,
            collate_fn=collate_with_pre_batching,
            sampler=sampler)

        dataloader.sampler.set_epoch(epoch=epoch)
        total_steps = len(dataloader)

        if self.c.verbose:
            print('Successfully loaded distributed batch dataloader.')

        return dataloader, total_steps

    def get_num_steps_per_epoch(self):
        if self.c.exp_batch_size == -1:
            return 1
        
        N = self.dataset.metadata['N']
        return int(np.ceil(N / self.c.exp_batch_size))

    def get_max_epochs(self):
        # When evaluating row interactions:
        # We assume a trained model loaded from checkpoint.
        # Run two epochs:
        #   - (1) evaluate train/val/test loss without row corruptions
        #   - (2) evaluate train/val/test loss with row corruptions

        num_steps_per_epoch = self.get_num_steps_per_epoch()
        return int(
            np.ceil(self.c.exp_num_total_steps / num_steps_per_epoch))

    def per_epoch_train_eval(self, epoch):
        early_stop = False
        if self.c.verbose:
            print(f'Epoch: {epoch}/{self.max_epochs}.')

        # need to increase step counter by one here (because step counter is)
        # still at last step
        end_experiment = (
                self.scheduler.num_steps + 1 >= self.c.exp_num_total_steps)

        eval_model = ((end_experiment or self.eval_check(epoch)) and
                      not self.c.model_is_semi_supervised)
        # The returned train loss is used for logging at eval time
        # It is None if minibatch_sgd is enabled, in which case we
        # perform an additional forward pass over all train entries
        if self.print:
            print("running training epoch: {}".format(epoch))
        train_loss = self.run_epoch(dataset_mode='train', epoch=epoch,
                                        eval_model=False)

        if eval_model:
            early_stop = self.eval_model(
                train_loss, epoch, end_experiment)
        if early_stop or end_experiment:
            early_stop = True
            return early_stop

        return early_stop

    def train_and_eval(self):
        """Main training and evaluation loop."""

        if self.is_distributed and self.c.mp_no_sync != -1:
            curr_epoch = 1

            while curr_epoch <= self.max_epochs:
                if self.print:
                    print(f'Current epoch: {curr_epoch}')
                with self.model.no_sync():
                    if self.print:
                        print(f'No DDP synchronization for the next '
                              f'{self.c.mp_no_sync} epochs.')
                    
                    for epoch in range(
                            curr_epoch, curr_epoch + self.c.mp_no_sync):
                        if self.per_epoch_train_eval(epoch=epoch):
                            return

                        if epoch >= self.max_epochs:
                            sys.exit(1)
                            return

                curr_epoch += self.c.mp_no_sync
                if (curr_epoch == self.c.exp_ssl_epochs and 
                self.c.model_is_semi_supervised):
                    if self.print:
                        print('stop self supervised')
                    self.torch_dataset.stop_selfsupervised(self.dataset.cv_dataset)

                if epoch >= self.max_epochs:
                    return
                if self.print:
                    print(f'Synchronizing DDP gradients in this epoch '
                          f'(epoch {curr_epoch}).')
                if self.per_epoch_train_eval(epoch=curr_epoch):
                    return

                curr_epoch += 1
        else:
            for epoch in range(1, self.max_epochs + 1):
                _ = self.per_epoch_train_eval(epoch=epoch)
                if epoch == self.max_epochs + 1:
                    break

    def eval_model(self, train_loss, epoch, end_experiment, return_dicts:bool=False):
        """Obtain val and test losses."""
        kwargs = dict(epoch=epoch, eval_model=True)
       
        val_loss_writer = []
        for recon in range(self.c.exp_num_reconstruction):
            if (self.is_distributed and self.rank==0) or (not self.is_distributed):
                to_print= f'recon: {recon+1}/{self.c.exp_num_reconstruction:}'
                print(f'\n{to_print:#^80}\n')
            val_loss = self.run_epoch(dataset_mode='val', **kwargs)
            val_log_loss = (self.loss.val_loss_logg['loss_val_epoch'] / 
                            self.loss.val_loss_logg['num_val_pred'])
                
            val_loss_writer.append(val_log_loss)
            self.dataset.cv_dataset.rec_count += 1
            self.loss.normalize_and_finalize_loss()
            self.loss.reset_logs()
            self.dataset.cv_dataset.reset_valnum()
            
        self.dataset.cv_dataset.reset_count()
        
        if (self.is_distributed and self.rank==0) or (not self.is_distributed):
            val_loss_writer = torch.mean(torch.cat(val_loss_writer))
            print('Validation Loss:', val_loss_writer)
            self.writer.add_scalar('Validation/val_loss', val_loss_writer.item(), epoch)
            
        self.val_dict = dict()

        if self.c.exp_normalize_ad_loss:
            if not self.c.exp_aggregation == 'sum':
                for key, item in self.loss.normalized_loss_val.items():
                    self.val_dict[key] = torch.max(torch.stack(item, dim=0))
            else:
                for key, item in self.loss.normalized_loss_val.items():
                    self.val_dict[key] = torch.sum(torch.stack(item, dim=0))
        else:
            if not self.c.exp_aggregation == 'sum':
                for key, item in self.loss.loss_val.items():
                    self.val_dict[key] = torch.max(torch.stack(item, dim=0))
            else:
                for key, item in self.loss.loss_val.items():
                    self.val_dict[key] = torch.sum(torch.stack(item, dim=0))
            
        self.loss.reset_val_loss()
        
        if self.is_distributed:
            data_save_dir = os.path.join('./results', 
                                         self.c.data_set,)
            if not os.path.isdir(data_save_dir) and self.rank==0:
                    os.mkdir(data_save_dir)

            ## since dictionnary gathering for multigpu is still not handled
            ## by pytorch, we save dictionnaries for each gpu for metric
            ## for metric computation. Ugly but works.
                    
            save_path = os.path.join('./results', 
                                        self.c.data_set, 
                                        self.c.res_dir)

            if not os.path.isdir(save_path) and self.rank==0:
                os.mkdir(save_path)
            dist.barrier()
                
            save_path = os.path.join(save_path, 
                                    f'seed_{self.c.torch_seed}_dict_rank_{self.rank}_epoch_{epoch}.pt')
            
            self.val_dict = {key: item.cpu() for key, item 
                                    in self.val_dict.items()}
            
            torch.save(self.val_dict, save_path)
            
            dist.barrier()
            
            if self.rank == 0:
                (max_metric_dict, 
                ratio_metric_dict) = self.compute_ad_metrics(epoch, 
                                                            aggregation=self.c.exp_aggregation,)
                
                
        else:
            (max_metric_dict, 
                ratio_metric_dict) = self.compute_ad_metrics(epoch, 
                                                            aggregation=self.c.exp_aggregation,)
        if return_dicts:
            if not self.is_distributed or self.rank==0:
                return (ratio_metric_dict, max_metric_dict)
            else:
                return (None, None)
        else:
            return False

    def run_epoch(self, dataset_mode, epoch, eval_model=False):
        """Train or evaluate model for a full epoch.

        Args:
            dataset_mode (str) {'train', 'test', 'eval'}: Depending on value
                mask/input the relevant parts of the data.
            epoch (int): Only relevant for logging.
            eval_model (bool): If this is true, write some extra metrics into
                the loss_dict. Is always true for test and eval, but only
                sometimes true for train. (We do not log each train epoch).

        Returns:
            loss_dict: Results of model for logging purposes.

        If `self.c.exp_minibatch_sgd` is True, we backpropagate after every
        mini-batch. If it is False, we backpropagate once per epoch.
        """
        print_n = self.c.exp_print_every_nth_forward

        # Model prep
        # We also want to eval train loss
        if (dataset_mode == 'train') and not eval_model:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        # Dataset prep -- prepares dataset.batch_gen attribute
        # Relevant in 'production' setting: we want to only input train
        # at train, train/val at val and train/val/test at test.
        self.dataset.set_mode(mode=dataset_mode, epoch=epoch)

        # Initialize data loaders (serial / distributed, pinned memory)
        if self.is_distributed:
            # TODO: parallel DDP loading?
            self.torch_dataset.materialize(cv_dataset=self.dataset.cv_dataset)
            batch_iter, num_batches = self.get_distributed_dataloader(epoch)
        else:
            # TODO: can be beneficial to test > cpu_count() procs if our
            # loading is I/O bound (which it probably is)
            batch_dataset = self.dataset.cv_dataset
            extra_args = {}

            if not self.c.data_set_on_cuda:
                extra_args['pin_memory'] = True

            batch_iter = torch.utils.data.DataLoader(
                dataset=batch_dataset,
                batch_size=1,  # The dataset is already batched.
                shuffle=False,  # Already shuffled
                num_workers=self.data_loader_nprocs,
                collate_fn=collate_with_pre_batching,
                **extra_args)
            batch_iter = tqdm(
                batch_iter, desc='Batch') if self.c.verbose else batch_iter

        for batch_index, batch_dict_ in enumerate(batch_iter):
            if ( (not hasattr(self,'old_indice_to_target')) and 
                ('old_indice_to_target' in list(batch_dict_.keys())) ):
                self.old_indice_to_target = batch_dict_['old_indice_to_target']

            self.run_batch(
                batch_dict_, dataset_mode, eval_model,
                epoch, print_n, batch_index)


        # Perform batch GD?
        batch_GD = (dataset_mode == 'train') and (
            not self.c.exp_minibatch_sgd)

        if eval_model or batch_GD:
            # We want loss_dict either for logging purposes
            # or to backpropagate if we do full batch GD
            loss_dict, loss_val = self.loss.finalize_epoch_losses(eval_model)

        # (See docstring) Either perform full-batch GD (as here)
        # or mini-batch SGD (in run_batch)
        if (not eval_model) and batch_GD:
            # Backpropagate on the epoch loss
            train_loss = loss_dict['total_loss']
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()
            self.optimizer.zero_grad()

        # Reset batch and epoch losses
        # loss.epoch_loss and loss.batch_loss are put back to zero
        # this is done for next epoch
        self.loss.reset()

        # Always return loss_dict
        # - If we are doing minibatching, return None to signify we must
        #       perform another set of mini-batch forward passes over train
        #       entries to get an eval loss.
        # - If we are doing full-batch training, we return the loss dict to
        #       immediately report loss metrics at eval time.
        if (not eval_model) and self.c.exp_minibatch_sgd:
            loss_dict = None

        return loss_dict

    def run_batch(self, batch_dict, dataset_mode, eval_model,
                  epoch, print_n, batch_index):
        # In stochastic label masking, we actually have a separate
        # label_mask_matrix. Else, it is just None.
        
        masked_tensors, label_mask_matrix, augmentation_mask_matrix = (
            batch_dict['masked_tensors'],
            batch_dict['label_mask_matrix'],
            batch_dict['augmentation_mask_matrix'])

        # Construct ground truth tensors
        ground_truth_tensors = batch_dict['data_arrs']

        # val_batch size
        val_batchsize = batch_dict.get('num_val')

        if not self.c.data_set_on_cuda:
            if self.is_distributed:
                device = self.gpu
            else:
                device = self.c.exp_device

            # non_blocking flag is appropriate when we are pinning memory
            # and when we use Distributed Data Parallelism

            # If we are fitting the full dataset on GPU, the following
            # tensors are already on the remote device. Otherwise, we can
            # transfer them with the non-blocking flag, taking advantage
            # of pinned memory / asynchronous transfer.

            # Cast tensors to appropriate data type
            ground_truth_tensors = [
                torch_cast_to_dtype(obj=data, dtype_name=self.c.data_dtype)
                for data in ground_truth_tensors]
            ground_truth_tensors = [
                data.to(device=device, non_blocking=True)
                for data in ground_truth_tensors]
            masked_tensors = [
                data.to(device=device, non_blocking=True)
                for data in masked_tensors]

            # Send everything else used in loss compute to the device
            batch_dict[f'{dataset_mode}_mask_matrix'] = (
                batch_dict[f'{dataset_mode}_mask_matrix'].to(
                    device=device, non_blocking=True))

            if augmentation_mask_matrix is not None:
                augmentation_mask_matrix = augmentation_mask_matrix.to(
                    device=device, non_blocking=True)

            # Need label_mask_matrix for stochastic label masking
            if label_mask_matrix is not None:
                if isinstance(label_mask_matrix, np.ndarray):
                    label_mask_matrix = torch.tensor(label_mask_matrix)
                    label_mask_matrix = label_mask_matrix.to(device=device, non_blocking=True)
                else:
                    label_mask_matrix = label_mask_matrix.to(device=device, non_blocking=True)

        forward_kwargs = dict(
            batch_dict=batch_dict,
            ground_truth_tensors=ground_truth_tensors,
            masked_tensors=masked_tensors, dataset_mode=dataset_mode,
            eval_model=eval_model, epoch=epoch,
            label_mask_matrix=label_mask_matrix,
            augmentation_mask_matrix=augmentation_mask_matrix,
            val_batchsize=val_batchsize)

        # This Automatic Mixed Precision autocast is a no-op
        # of c.model_amp = False
        with torch.cuda.amp.autocast(enabled=self.c.model_amp):
            self.forward_and_loss(**forward_kwargs)

        # (See docstring) Either perform mini-batch SGD (as here)
        # or full-batch GD (as further below)
        if (dataset_mode == 'train' and self.c.exp_minibatch_sgd
                and (not eval_model)):
            # Standardize and backprop on minibatch loss
            # if minibatch_sgd enabled
            loss_dict = self.loss.finalize_batch_losses()
            #print(loss_dict)
            train_loss = loss_dict['total_loss']

            if (self.is_distributed and self.rank==0) or (not self.is_distributed):
                self.writer.add_scalar('Train/Total_loss', train_loss.item(), self.n_iter)

            # ### Apply Automatic Mixed Precision ###
            # The scaler ops will be no-ops if we have specified
            # c.model_amp is False in the Trainer init

            # Scales loss.
            # Calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(train_loss).backward()

            # scaler.step() first unscales the gradients of the
            # optimizer's assigned params.
            # If these gradients do not contain infs or NaNs,
            # optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()
            self.n_iter += 1

            self.scheduler.step()
            self.optimizer.zero_grad()

        # Update the epoch loss info with detached minibatch losses
        self.loss.update_losses(eval_model=eval_model,)

    def forward_and_loss(
            self, batch_dict, ground_truth_tensors, masked_tensors,
            dataset_mode, eval_model, epoch, label_mask_matrix,
            augmentation_mask_matrix, ad:bool=False, 
            val_batchsize:int=1):
        """Run forward pass and evaluate model loss."""
        extra_args = {}
        
        if eval_model:
            with torch.no_grad():
                output = self.model(masked_tensors, **extra_args)
        else:
            output = self.model(masked_tensors, **extra_args)

        loss_kwargs = dict(
            output=output, ground_truth_data=ground_truth_tensors,
            label_mask_matrix=label_mask_matrix,
            augmentation_mask_matrix=augmentation_mask_matrix,
            data_dict=batch_dict, dataset_mode=dataset_mode,
            eval_model=eval_model, val_batchsize=val_batchsize)

        # By doing loss.compute, the loss.batch_loss is replaced by its
        # new value. It will serve in up loss.update_losses() to be
        # added to loss.epoch_loss
        self.loss.compute(**loss_kwargs)

    def eval_check(self, epoch):
        """Check if it's time to evaluate val and test errors."""

        if self.c.exp_eval_every_epoch_or_steps == 'epochs':
            return epoch % self.c.exp_eval_every_n == 0
        elif self.c.exp_eval_every_epoch_or_steps == 'steps':
            # Cannot guarantee that we hit modulus directly.
            if (self.scheduler.num_steps - self.last_eval >=
                    self.c.exp_eval_every_n):
                self.last_eval = self.scheduler.num_steps
                return True
            else:
                return False
        else:
            raise ValueError

    def compute_ad_metrics(self, epoch:int=None, 
                           return_preds:bool=False,
                           aggregation:str='sum',):
        

        # align target dict and score dict
        target_dict = dict(sorted(self.dataset.cv_dataset.old_indice_to_target_val.items()))
        
        filename = 'seed_{seed}_dict_rank_{rank}_epoch_{epoch}.pt'
        
        if self.is_distributed:
            for rank in range(self.world_size):
                current_filename = filename.format(seed=self.c.torch_seed,
                                                   rank=rank,
                                                   epoch=epoch)
                load_path = os.path.join('./results',
                                         self.c.data_set,
                                         self.c.res_dir,
                                         current_filename)
                if rank == 0:
                    val_score_dict = torch.load(load_path)
                else:
                    _ = torch.load(load_path)
                    val_score_dict.update(_)
            val_score = dict(sorted(val_score_dict.items()))
            
            _filename_ = 'seed_{seed}_epoch_{epoch}.pt'
            current_filename = _filename_.format(seed=self.c.torch_seed,
                                                 epoch=epoch)
            save_path_ = os.path.join('./results',
                                      self.c.data_set,
                                      self.c.res_dir,
                                      current_filename)
            torch.save(val_score, save_path_)
            
            rm_name = 'seed_{seed}_dict_rank_{rank}_epoch_*'
            rm_name = rm_name.format(seed=self.c.np_seed,
                                     rank=rank,)
            rm_path = os.path.join('./results',
                                   self.c.data_set,
                                   self.c.res_dir,
                                   rm_name)
            for file in glob.glob(rm_path):
                os.remove(file)
            
        else:
            val_score = dict(sorted(self.val_dict.items()))
            val_score = {key: item.cpu() for key, item in val_score.items()}
            target_dict = dict(sorted(self.dataset.cv_dataset.old_indice_to_target_val.items()))
        
        target_array = np.vstack(list(target_dict.values())).flatten()
        target_tensors = torch.as_tensor(target_array.astype(np.float),
                                         dtype=torch.int8).squeeze()
                
        val_tensors = torch.stack(list(val_score.values())).squeeze()
        
        pr_curve = BinaryPrecisionRecallCurve(thresholds=None)
        avg_prec = AveragePrecision(task="binary", pos_label=1)
        auroc = AUROC(task="binary", pos_label=1)
        
        #normalize val_score
        val_tensors = ((val_tensors - torch.mean(val_tensors)) 
                       / torch.std(val_tensors))

        precisions, recalls, thresholds = pr_curve(val_tensors,
                                                   target_tensors)
        ap = avg_prec(val_tensors, target_tensors)
        auc = auroc(val_tensors, target_tensors)

        f1s = 2 * (precisions * recalls) / (precisions + recalls)
        f1s = torch.nan_to_num(f1s, nan=0.)
        
        idx_best_f1 = torch.argmax(f1s)
        best_f1 = f1s[idx_best_f1]
        thresh = thresholds[idx_best_f1]

        max_metric_dict = {'seed': self.c.torch_seed,
                           'F1':best_f1,
                           'ap':ap,
                           'auc': auc,}

        print('F1 Score with the thereshold that maximizes it', max_metric_dict)
        
        thresh = np.percentile(val_tensors.numpy(), self.c.ratio)
        target_pred = (val_tensors.numpy() >= thresh).astype(int)
        target_true = target_tensors.numpy().astype(int)
        (_, _, f_score_ratio, _) = prf(target_true, target_pred, average='binary')
        
        ratio_metric_dict = {'seed': self.c.torch_seed,
                             'F1':f_score_ratio,
                             'ap':ap,
                             'auc': auc,}
        
        print('ratio F1 Score', ratio_metric_dict)
        
            
        max_outname = (f'results_maxf1_'
                         f'epoch_{epoch}__'
                         f'seed_{self.c.torch_seed}.pkl')
        ratio_outname = (f'results_ratiof1_'
                         f'epoch_{epoch}__'
                         f'seed_{self.c.torch_seed}.pkl')
            
        result_dataset_path = os.path.join('./results', self.c.data_set,)
        if not os.path.isdir(result_dataset_path):
            os.mkdir(result_dataset_path)
        folder_path = os.path.join('./results', self.c.data_set, self.c.res_dir)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        for name, metric_dict in zip([max_outname, ratio_outname],
                              [max_metric_dict, ratio_metric_dict]):
        
            save_path = os.path.join(folder_path, name)
            with open(save_path, 'wb') as f:
                pickle.dump(metric_dict, f)

        out = ((max_metric_dict, ratio_metric_dict),
               target_pred) if return_preds else (max_metric_dict, ratio_metric_dict)
        return out
