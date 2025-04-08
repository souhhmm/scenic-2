import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.dataset_name = 'audiovisual_tfrecord_dataset'
    # Dataset configuration
    config.dataset_configs = ml_collections.ConfigDict()
    config.dataset_configs.base_dir = 'scenic/projects/mbt/sample_tfrecords'  # Directory containing your TFRecords
    config.dataset_configs.tables = {
        'train': 'train@5',
        'validation': 'train@5',  # Using 5 shards
        'test': 'train@5'
    }
    config.dataset_configs.examples_per_subset = {
        'train': 100,  # Total number of examples 
        'validation': 100,
        'test': 100
    }
    config.dataset_configs.num_classes = 60
    config.data_dtype_str = 'float32'
    
    # Add these required fields
    config.dataset_configs.modalities = ('rgb', 'spectrogram')
    config.dataset_configs.return_as_dict = True
    config.dataset_configs.num_frames = 32
    config.dataset_configs.stride = 2
    config.dataset_configs.num_spec_frames = 5
    config.dataset_configs.spec_stride = 1
    config.dataset_configs.min_resize = 256
    config.dataset_configs.crop_size = 224
    config.dataset_configs.spec_shape = (100, 128)
    config.dataset_configs.one_hot_labels = True
    config.dataset_configs.zero_centering = True
    
    # Multicrop eval settings
    config.dataset_configs.do_multicrop_test = True
    config.dataset_configs.log_test_epochs = 4
    config.dataset_configs.num_test_clips = 4
    config.dataset_configs.test_batch_size = 8  # Needs to be num_local_devices
    config.multicrop_clips_per_device = 2
    
    # Augmentation parameters
    config.dataset_configs.augmentation_params = ml_collections.ConfigDict()
    config.dataset_configs.augmentation_params.do_jitter_scale = True
    config.dataset_configs.augmentation_params.scale_min_factor = 0.9
    config.dataset_configs.augmentation_params.scale_max_factor = 1.33
    config.dataset_configs.augmentation_params.prob_scale_jitter = 1.0
    config.dataset_configs.augmentation_params.do_color_augment = True
    config.dataset_configs.augmentation_params.prob_color_augment = 0.8
    config.dataset_configs.augmentation_params.prob_color_drop = 0.1
    
    config.dataset_configs.prefetch_to_device = 2
    
    # SpecAugment hyperparameters
    config.dataset_configs.spec_augment = True
    config.dataset_configs.spec_augment_params = ml_collections.ConfigDict()
    config.dataset_configs.spec_augment_params.freq_mask_max_bins = 48
    config.dataset_configs.spec_augment_params.freq_mask_count = 1
    config.dataset_configs.spec_augment_params.time_mask_max_frames = 48
    config.dataset_configs.spec_augment_params.time_mask_count = 4
    config.dataset_configs.spec_augment_params.time_warp_max_frames = 1.0
    config.dataset_configs.spec_augment_params.time_warp_max_ratio = 0
    config.dataset_configs.spec_augment_params.time_mask_max_ratio = 0
    
    # Model configuration
    config.model_name = 'mbt_classification'
    config.model = ml_collections.ConfigDict()
    config.model.hidden_size = 768
    config.model.patches = ml_collections.ConfigDict()
    config.model.patches.size = [16, 16]
    config.model.num_heads = 12
    config.model.mlp_dim = 3072
    config.model.num_layers = 12
    config.model.attention_dropout_rate = 0.0
    config.model.dropout_rate = 0.1
    
    # Add these required model fields
    config.model.modality_fusion = ('rgb', 'spectrogram')
    config.model.use_bottleneck = True
    config.model.test_with_bottlenecks = True
    config.model.share_encoder = False
    config.model.n_bottlenecks = 4
    config.model.fusion_layer = 8
    
    # Training configuration
    config.batch_size = 32
    config.eval_batch_size = 32
    config.rng_seed = 0
    config.trainer_name = 'mbt_trainer'
    
    # Additional training parameters
    config.optimizer = 'momentum'
    config.optimizer_configs = ml_collections.ConfigDict()
    config.l2_decay_factor = 0
    config.max_grad_norm = 1
    config.label_smoothing = 0.3
    config.num_training_epochs = 50
    
    # Mixup configuration
    config.mixup = ml_collections.ConfigDict()
    config.mixup.alpha = 0.5
    config.mixmod = False
    
    # Additional regularization
    config.model.stochastic_droplayer_rate = 0.3

    # ImageNet checkpoint configuration
    config.init_from = ml_collections.ConfigDict()
    config.init_from.checkpoint_path = 'scenic/projects/mbt/ViT_B_16_ImageNet21k'  # Update this path
    config.init_from.checkpoint_format = 'scenic'  # or 'big_vision'
    config.init_from.model_config = ml_collections.ConfigDict()
    config.init_from.model_config.model = ml_collections.ConfigDict()
    config.init_from.model_config.model.classifier = 'token'  # or 'gap'
    config.init_from.restore_positional_embedding = True
    config.init_from.restore_input_embedding = True
    config.init_from.positional_embed_size_change = 'resize_tile'
    
    return config