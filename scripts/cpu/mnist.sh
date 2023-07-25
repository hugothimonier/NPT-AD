#!/bin/bash

conda activate nptad
rm -rf ./data/mnist/mnist/*
export Q_DATE=$(date "+%d_%m_%y_%H_%M") 
mkdir -p ./logs/mnist
mkdir -p ./results/mnist

python -u run.py \
    --n_runs 20 \
    --data_path ./data/mnist/ \
    --data_set mnistad \
    --res_dir job_$Q_DATE \
    --data_loader_nprocs 0 \
    --data_set_on_cuda False \
    --exp_name mnist_params \
    --np_seed 1 \
    --torch_seed 1 \
    --original_torch_seed 1 \
    --exp_batch_size -1 \
    --exp_val_batchsize 400 \
    --exp_num_train_inference -1 \
    --exp_num_total_steps 1000 \
    --exp_eval_every_n 1000 \
    --exp_eval_every_epoch_or_steps epochs \
    --exp_optimizer lookahead_lamb \
    --exp_lookahead_update_cadence 6 \
    --exp_optimizer_warmup_proportion 0 \
    --exp_optimizer_warmup_fixed_n_steps 100 \
    --exp_lr 0.001 \
    --exp_scheduler flat_and_anneal \
    --exp_gradient_clipping 1.0 \
    --exp_weight_decay 0 \
    --exp_deterministic_masks True \
    --exp_checkpoint_setting all_checkpoints \
    --exp_ssl_epochs 0 \
    --exp_normalize_ad_loss False \
    --exp_cat_as_num_features False \
    --exp_keep_categorical_features True \
    --model_init_weights True \
    --model_init_params 0.05 \
    --model_init_params 0 \
    --model_init_type xavier \
    --mp_no_sync -1 \
    --mp_bucket_cap_mb -1 \
    --exp_aggregation sum \
    --model_class NPT \
    --model_is_semi_supervised False \
    --model_amp True \
    --model_feature_type_embedding False \
    --model_feature_index_embedding True \
    --model_bert_augmentation True \
    --model_bert_mask_percentage 0.9 \
    --model_augmentation_bert_mask_prob "dict(train=0.15, val=0.3, test=0.15)" \
    --model_embedding_layer_norm True \
    --model_att_block_layer_norm True \
    --model_layer_norm_eps 1e-12 \
    --model_pre_layer_norm True \
    --model_hidden_dropout_prob 0.1 \
    --model_att_score_dropout_prob 0.1 \
    --model_checkpoint_key job__$Q_DATE \
    --model_dim_hidden 32 \
    --model_num_heads 4 \
    --model_sep_res_embed True \
    --model_stacking_depth 4\
    --model_mix_heads True \
    --model_rff_depth 1 \
    --ad True \
    --data_force_reload True > ./logs/mnist/$Q_DATE.log