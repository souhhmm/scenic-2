import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.dataset_name = 'sample_dataset'
    # Dataset configuration
    config.dataset_configs = ml_collections.ConfigDict()
    config.dataset_configs.base_dir = 'scenic/projects/mbt/sample_tfrecords'  # Directory containing your TFRecords
    config.dataset_configs.tables = {
        'train': 'train@5'  # Using 5 shards
    }
    config.dataset_configs.examples_per_subset = {
        'train': 100  # Total number of examples    
    }
    config.dataset_configs.num_classes = 60
    
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
    config.trainer_name = 'mbt_trainer'  # Add this

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