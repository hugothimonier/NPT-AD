"""Define argument parser."""
import argparse

DEFAULT_AUGMENTATION_BERT_MASK_PROB = {
    'train': 0.15,
    'val': 0.,
    'test': 0.
}


def build_parser():
    """Build parser."""
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    ###########################################################################
    # #### Data Config ########################################################
    ###########################################################################
    parser.add_argument(
        '--res_dir', type=str, default='./',
        help='Revelevant for distributed training. Name of directory where '
             ' the dictionnaries containing the anomaly scores.')
    parser.add_argument(
        '--data_path', type=str, default='data',
        help='Path of data')
    parser.add_argument(
        '--data_set', type=str, default='abalone',
        help='accepted values are currently: '
            'abalone, forestcoverad, mammography, satellite, vertebral, '
            'annthyroid, glass, mnistad, satimage, vowels, '
            'arrhythmia, ionosphere, mulcross, seismic, wbc, '
            'kdd, musk, separable, wine, '
            'breastw, kddrev, optdigits, shuttle, '
            'cardio, letter, pendigits, speech, '
            'ecoli, lympho, pima, thyroid.')
    parser.add_argument(
        '--data_loader_nprocs', type=int, default=0,
        help='Number of processes to use in data loading. Specify -1 to use '
             'all CPUs for data loading. 0 (default) means only the main  '
             'process is used in data loading. Applies for serial and '
             'distributed training.')
    parser.add_argument(
        '--data_set_on_cuda', type='bool', default=False,
        help='Place the entire dataset and metadata necessary per epoch on '
             'the CUDA device. Appropriate for smaller datasets.')
    parser.add_argument(
        '--data_force_reload', default=False, type='bool',
        help='If True, reload CV splits, ignoring cached files.')
    parser.add_argument(
        '--data_log_mem_usage', default=False, action='store_true',
        help='If True, report mem size of full dataset, as loaded from cache.')
    parser.add_argument(
        '--data_clear_tmp_files', type='bool', default=False,
        help=f'If True, deletes all downloaded/unzipped files in the dataset '
             f'folder while the CV split is being materialized and cached '
             f'(e.g. necessary to keep Higgs footprint under 30GB on Azure).')
    parser.add_argument(
        '--data_dtype',
        default='float32',
        type=str, help='Data type (supported for float32, float64) '
                         'used for data (e.g. ground truth, masked inputs).')
    parser.add_argument(
        '--verbose', 
        default=False,
        type='bool'
    )

    ###########################################################################
    # #### Experiment Config ##################################################
    ###########################################################################
    parser.add_argument(
        '--exp_device', default=None, type=str,
        help='If provided, use this (CUDA) device for the run.')
    parser.add_argument(
        '--exp_name', type=str, default=None,
        help='Give experiment a name.')
    parser.add_argument(
        '--np_seed', type=int, default=42,
        help='Random seed for numpy. Set to -1 to choose at random.')
    parser.add_argument(
        '--torch_seed', type=int, default=42,
        help='Random seed for torch. Set to -1 to choose at random.')
    parser.add_argument(
        '--exp_disable_cuda', dest='exp_use_cuda', default=True,
        action='store_false', help='Disable GPU acceleration')
    parser.add_argument(
        '--exp_n_runs', type=int, default=1,
        help=f'Maximum number of CV runs. This upper bounds the number of '
             f'cross validation folds that are being executed.')
    parser.add_argument(
        '--exp_batch_size', type=int, default=-1,
        help='Number of instances (rows) in each batch '
             'taken as input by the model. -1 corresponds to no '
             'minibatching.')
    parser.add_argument(
        '--exp_full_batch_gd', dest='exp_minibatch_sgd',
        default=True, action='store_false',
        help='Full batch gradient descent (as opposed to mini-batch GD)')
    parser.add_argument(
        '--exp_val_perc', type=float, default=0.1,
        help='Percent of examples in validation set')
    
    ## <NPT-AD>
    parser.add_argument(
        '--exp_val_batchsize', type=int, default=1,
        help='Number of validation samples to feed for the reconstruction '
             'in the matrix for AD. Recall that X = [val_samples, '
             ' subsample of train]^T. Right now only 1 is supported, '
             'TODO: I shall check later how to handle multi gpu in that case, '
             ' e.g. one val sample per gpu. Might be complicated to handle.')
    parser.add_argument(
        '--exp_num_train_inference', type=int, default=0,
        help='Number of train samples to feed for the reconstruction '
             'in the matrix for AD. Recall that X = [val_samples, '
             ' subsample of train]^T. -1 Corresponds to the whole train'
             'set. Mostly used for small datasets'
             '(e.g. arrhythmia, thyroid')
    parser.add_argument(
        '--exp_keep_categorical_features', type='bool', default=True,
        help='Whether to keep categorical features or exclude them.')
    parser.add_argument(
        '--exp_cat_as_num_features', type='bool', default=True,
        help='Whether to use categorical features as numerical features.')
    parser.add_argument(
        '--exp_num_reconstruction', type=int, default=15,
        help='Number of reconstruction and masking to perform to compute '
             'average reconstruction error as an anomaly score. Depreciated.')
    parser.add_argument(
        '--exp_deterministic_masks', type='bool', 
        default=False,
        help='Whether to use masking for each feature.')
    parser.add_argument(
        '--exp_n_hidden_features', action='append', type=int,
        default=[1],
        help='Number of features to mask at a time.'
             'Determines the number of reconstruction.')
    parser.add_argument(
        '--exp_max_n_recon', type=int,
        default=0,
        help='Maximum number of recon in case of deterministic'
             ' mask.')
    parser.add_argument(
        '--exp_aggregation', type=str, default='sum', choices={'max', 'sum'},
        help='Type of reconstruction aggregation. Must be none or sum.'
             ' Valid only for AD.')
    parser.add_argument(
        '--exp_ssl_epochs', type=float, default=None,
        help='Number of epochs for which validation set is included'
        'in the training set. Used only if model_is_semi_supervised is True.')
    parser.add_argument(
        '--exp_contamination_share_train', type=float, default=0,
        help='When --ad is True, allows for a portion of the training set to be'
        'contaminated with a portion of the anomalies defined here. Depreciated.')
    parser.add_argument(
        '--share_contamination', type=float, default=0.,
        help='Anomaly share in training set')
    parser.add_argument(
        '--exp_normalize_ad_loss', type='bool', default=False,
        help='Whether to normalize each reconstruction loss in the process'
        'of construction the anomaly score.')
    parser.add_argument(
        '--anomalies_in_inference', type='bool', default=False,
        help='Whether to use as samples to reconstruct only anomalies'
        'Works only for separable dataset.')
    parser.add_argument(
        '--model_init_weights', type='bool', default=False,
        help='Init the weights of the network or not')
    parser.add_argument(
        '--model_init_type', type=str, default='xavier',
        help='Define which initialization type. Choice is xavier or'
        'normal.')
    parser.add_argument(
        '--model_init_params', action='append',
        help='Parameters for normal initialization.')
    parser.add_argument(
        '--original_np_seed', type=int, default=91,
        help='original seed for numpy seed iterations.'
        'If n_runs==1, sets the numpy seed for the run. '
        'If n_runs > 1, sets the first run seed, each new '
        'run will add 1 to the chosen seed.')
    parser.add_argument(
        '--original_torch_seed', type=int, default=91,
        help='original seed for torch seed iterations. '
        'If n_runs==1, sets the torch seed for the run. '
        'If n_runs > 1, sets the first run seed, each new '
        'run will add 1 to the chosen seed.')
    parser.add_argument(
        '--n_runs', type=int, default=1,)
    parser.add_argument(
        '--exp_mix_rows_inference', type='bool',
        help='Mix rows of unmasked training samples to measure the '
        'sample-sample dependency effect.', default=False,)
    ## </NPT-AD>
        
    parser.add_argument(
        '--ad', type='bool', default=False,
        help='Anomaly detection set-up')
    parser.add_argument(
        '--exp_test_perc', type=float, default=0.2,
        help='Percent of examples in test set. '
             'Determines number of CV splits.')
    parser.add_argument(
        '--exp_num_total_steps', type=float, default=100e3,
        help='Number of total gradient descent steps. The maximum number of '
             'epochs is computed as necessary using this value (e.g. in '
             'gradient syncing across data parallel replicates in distributed '
             'training).')
    parser.add_argument(
        '--exp_patience', type=int, default=-1,
        help='Early stopping -- number of epochs that '
             'validation may not improve before training stops. '
             'Turned off by default.')
    parser.add_argument(
        '--exp_checkpoint_setting', type=str, default='best_model',
        help='Checkpointing -- determines if there is no checkpointing '
             '(None), only the best model thus far is cached '
             '(best_model), or all models that improve on val loss should be '
             'maintained (all_checkpoints). See npt/utils/eval_utils.py.')
    parser.add_argument(
        '--exp_cache_cadence', type=int, default=1,
        help='Checkpointing -- we cache the model every `exp_cache_cadence` '
             'times that the validation loss improves since the last cache. '
             'Set this value to -1 to disable caching.')
    parser.add_argument(
        '--exp_checkpoint_save', type=int, default=100e3,
        help='Checkpointing -- we cache the model at chosen iter.')
    parser.add_argument(
        '--exp_load_from_checkpoint', default=False, type='bool',
        help='If True, attempt to load from checkpoint and continue training.')
    parser.add_argument(
        '--exp_print_every_nth_forward', dest='exp_print_every_nth_forward',
        default=False, type=int,
        help='Print during mini-batch as well for large epochs.')
    parser.add_argument(
        '--exp_eval_every_n', type=int, default=5,
        help='Evaluate the model every n steps/epochs. (See below).')
    parser.add_argument(
        '--exp_eval_every_epoch_or_steps', type=str, default='epochs',
        help='Choose whether we eval every n "steps" or "epochs".')
    parser.add_argument(
        '--exp_eval_test_at_end_only',
        default=False,
        type='bool',
        help='Evaluate test error only in last step.')

    # Optimization
    # -------------
    parser.add_argument(
        '--exp_optimizer', type=str, default='lookahead_lamb',
        help='Model optimizer: see npt/optim.py for options.')
    parser.add_argument(
        '--exp_lookahead_update_cadence', type=int, default=6,
        help='The number of steps after which Lookahead will update its '
             'slow moving weights with a linear interpolation between the '
             'slow and fast moving weights.')
    parser.add_argument(
        '--exp_optimizer_warmup_proportion', type=float, default=0.7,
        help='The proportion of total steps over which we warmup.'
             'If this value is set to -1, we warmup for a fixed number of '
             'steps. Literature such as Evolved Transformer (So et al. 2019) '
             'warms up for 10K fixed steps, and decays for the rest. Can '
             'also be used in certain situations to determine tradeoff '
             'annealing, see exp_tradeoff_annealing_proportion below.')
    parser.add_argument(
        '--exp_optimizer_warmup_fixed_n_steps', type=int, default=10000,
        help='The number of steps over which we warm up. This is only used '
             'when exp_optimizer_warmup_proportion is set to -1. See above '
             'description.')
    parser.add_argument(
        '--exp_lr', type=float, default=1e-3,
        help='Learning rate')
    parser.add_argument(
        '--exp_scheduler', type=str, default='flat_and_anneal',
        help='Learning rate scheduler: see npt/optim.py for options.')
    parser.add_argument(
        '--exp_gradient_clipping', type=float, default=1.,
        help='If > 0, clip gradients.')
    parser.add_argument(
        '--exp_weight_decay', type=float, default=0,
        help='Weight decay / L2 regularization penalty. Included in this '
             'section because it is set in the optimizer. '
             'HuggingFace default: 1e-5')

    ###########################################################################
    # #### Multiprocess Config ################################################
    ###########################################################################

    parser.add_argument(
        '--mp_distributed', dest='mp_distributed', default=False, type='bool',
        help='If True, run data-parallel distributed training with Torch DDP.')
    parser.add_argument(
        '--mp_nodes', dest='mp_nodes', default=1, type=int,
        help='number of data loading workers')
    parser.add_argument(
        '--mp_gpus', dest='mp_gpus', default=1, type=int,
        help='number of gpus per node')
    parser.add_argument(
        '--mp_nr', dest='mp_nr', default=0, type=int,
        help='ranking within the nodes')
    parser.add_argument(
        '--mp_no_sync', dest='mp_no_sync', default=-1, type=int,
        help='Number of forward pass iterations for which gradients are not '
             'synchronized. Increasing this number will result in a lower '
             'amortized cost of distribution (and hence, a closer-to-linear '
             'scaling of per-epoch time with respect to number of GPUs), '
             'at the cost of convergence stability.')
    parser.add_argument(
        '--mp_bucket_cap_mb', dest='mp_bucket_cap_mb', default=25, type=int,
        help='Larger values denote a larger gradient bucketing size for DDP. '
             'This reduces the amortized overhead of communication, but means '
             'that there is a longer lead time before reduction (i.e. '
             'AllReduce aggregating gradient buckets across GPUs). Larger '
             'models (e.g. BERT ~ 110M parameters) will likely benefit '
             'from mp_bucket_cap_mb in excess of 50 MB.')

    ###########################################################################
    # #### Model Class Config #################################################
    ###########################################################################

    parser.add_argument(
        '--model_class', dest='model_class', type=str,
        default='NPT',
        help='Specifies model(s) to train/evaluate.')

    ###########################################################################
    # #### General Model Config ###############################################
    ###########################################################################

    parser.add_argument(
        '--model_is_semi_supervised',
        default=True,
        type='bool', help='Include test features at training.')
    parser.add_argument(
        '--model_dtype',
        default='float32',
        type=str, help='Data type (supported for float32, float64) '
                       'used for model.')
    parser.add_argument(
        '--model_amp',
        default=False,
        type='bool', help='If True, use automatic mixed precision (AMP), '
                          'which can provide significant speedups on V100/'
                          'A100 GPUs.')
    parser.add_argument(
        '--model_feature_type_embedding', type='bool', default=True,
        help='When True, learn an embedding on whether each feature is '
             'numerical or categorical. Similar to the "token type" '
             'embeddings canonical in NLP. See https://github.com/huggingface/'
             'transformers/blob/master/src/transformers/models/bert/'
             'modeling_bert.py')
    parser.add_argument(
        '--model_feature_index_embedding', type='bool', default=True,
        help='When True, learn an embedding on the index of each feature '
             '(column). Similar to positional embeddings.')

    # #### Masking and Stochastic Forward Passes ##############################

    parser.add_argument(
        '--model_bert_augmentation', type='bool', default=True,
        help='When True, use BERT style augmentation. This introduces a mask '
             'column for each data entry, which can also be used to track '
             'values that are truly missing (i.e., for which there is no '
             'known ground truth). Set to False if: '
             '(i) You do not wish to use BERT augmentation'
             '(ii) You are confident there are no missing values in the '
             '      data.'
             '(iii) Given (i) and (ii), you do not want to include an '
             '      unneeded mask column for every entry.'
             'Note that you also must pass a model_augmentation_bert_mask_prob'
             ' dict with zeros for train, test, val.')
    parser.add_argument(
        '--model_bert_mask_percentage', type=float, default=0.9,
        help='Probability of actually masking out token after being selected.')
    parser.add_argument(
        '--model_augmentation_bert_mask_prob',
        type=str, default=DEFAULT_AUGMENTATION_BERT_MASK_PROB,
        help='Use bert style augmentation with the specified mask probs'
             'in training/validation/testing.')
    # Dicts can be passed as
    # --model_augmentation_bert_mask_prob "dict(train=.15, val=0, test=0)"

    # #### Normalization ######################################################

    parser.add_argument(
        '--model_embedding_layer_norm', default=False, type='bool',
        help='(Disable) use of layer normalization after in-/out-embedding.')
    parser.add_argument(
        '--model_att_block_layer_norm', default=True, type='bool',
        help='(Disable) use of layer normalization in attention blocks.')
    parser.add_argument(
        '--model_layer_norm_eps', default=1e-12, type=float,
        help='The epsilon used by layer normalization layers.'
             'Default from BERT.')
    parser.add_argument(
        '--model_att_score_norm', default='softmax', type=str,
        help='Normalization to use for the attention scores. Options include'
             'softmax, constant (which divides by the sqrt of # of entries).')
    parser.add_argument(
        '--model_pre_layer_norm', default=True, type='bool',
        help='If True, we apply the LayerNorm (i) prior to Multi-Head '
             'Attention, (ii) before the row-wise feedforward networks. '
             'SetTransformer and Huggingface BERT opt for post-LN, in which '
             'LN is applied after the residual connections. See `On Layer '
             'Normalization in the Transformer Architecture (Xiong et al. '
             '2020, https://openreview.net/forum?id=B1x8anVFPr) for '
             'discussion.')

    # #### Dropout ############################################################

    parser.add_argument(
        '--model_hidden_dropout_prob', type=float, default=0.1,
        help='The dropout probability for all fully connected layers in the '
             '(in, but not out) embeddings, attention blocks.')
    parser.add_argument(
        '--model_att_score_dropout_prob', type=float, default=0.1,
        help='The dropout ratio for the attention scores.')

    ###########################################################################
    # #### Attention Block Config ############################################
    ###########################################################################

    parser.add_argument(
        '--model_hybrid_debug',
        default=False,
        type='bool',
        help=f'Print dimensions of the input after each reshaping during '
             f'forward pass.')
    parser.add_argument(
        '--model_checkpoint_key',
        type=str,
        default=None,
        help=f'If provided, use as title of checkpoint subdirectory. Used to '
             f'avoid collisions between subtly different runs.')

    ###########################################################################
    # #### Multihead Attention Config #########################################
    ###########################################################################

    parser.add_argument(
        '--model_dim_hidden', type=int, default=64,
        help='Intermediate feature dimension.')
    parser.add_argument(
        '--model_num_heads', type=int, default=8,
        help='Number of attention heads. Must evenly divide model_dim_hidden.')
    parser.add_argument(
        '--model_sep_res_embed',
        default=True,
        type='bool',
        help='Use a seperate query embedding W^R to construct the residual '
        'connection '
        'W^R Q + MultiHead(Q, K, V)'
        'in the multi-head attention. This was not done by SetTransformers, '
        'which reused the query embedding matrix W^Q, '
        'but we feel that adding a separate W^R should only helpful.')
    parser.add_argument(
        '--model_stacking_depth',
        dest='model_stacking_depth',
        type=int,
        default=8,
        help=f'Number of layers to stack.')
    parser.add_argument(
        '--model_mix_heads',
        dest='model_mix_heads',
        type='bool',
        default=True,
        help=f'Add linear mixing operation after concatenating the outputs '
        f'from each of the heads in multi-head attention.'
        f'Set Transformer does not do this. '
        f'We feel that this should only help. But it also does not really '
        f'matter as the rFF(X) can mix the columns of the multihead attention '
        f'output as well. '
        f'model_mix_heads=False may lead to inconsistencies for dimensions.')
    parser.add_argument(
        '--model_rff_depth',
        dest='model_rff_depth',
        type=int,
        default=1,
        help=f'Number of layers in rFF block.')

    ###########################################################################
    # #### Visualization  #####################################################
    ###########################################################################

    parser.add_argument(
        '--viz_att_maps',
        default=False,
        type='bool',
        help=f'Using config settings, attempt to load most recent checkpoint '
             f'and produce attention map visualizations.')
    parser.add_argument(
        '--viz_att_maps_save_path',
        default='data/attention_maps',
        type=str,
        help=f'Save attention maps to file. Specify the save path here.')

    return parser


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/
    parsing-boolean-values-with-argparse/36031646"""
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")
