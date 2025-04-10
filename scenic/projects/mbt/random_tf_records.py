import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import cv2

def create_sample_tfrecords(output_dir, num_examples=100, num_shards=5):
    """Create sample TFRecords with random data for testing MBT."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate examples per shard
    examples_per_shard = num_examples // num_shards
    
    # Create shard writers
    for shard_idx in range(num_shards):
        shard_path = os.path.join(output_dir, f'train-{shard_idx:05d}-of-{num_shards:05d}')
        with tf.io.TFRecordWriter(shard_path) as writer:
            for example_idx in tqdm(range(examples_per_shard), desc=f'Creating shard {shard_idx}'):
                # Create random data
                # Video frames: (32, 256, 256, 3) normalized to [0, 1]
                frames = np.random.rand(32, 256, 256, 3).astype(np.float32)
                
                # Audio spectrogram: (100, 128) with values around -80 dB
                spectrogram = np.random.rand(100, 128).astype(np.float32) * 100 - 80
                
                # Random label between 0 and 59 (for 60 classes)
                label = np.random.randint(0, 60)
                
                # Create SequenceExample
                example = tf.train.SequenceExample()
                
                # Add video frames as JPEG-encoded images
                for frame in frames:
                    # Convert float32 [0,1] to uint8 [0,255]
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    # Encode as JPEG
                    _, jpeg_bytes = cv2.imencode('.jpg', frame_uint8)
                    # Add to sequence
                    example.feature_lists.feature_list['image/encoded'].feature.add().bytes_list.value.append(jpeg_bytes.tobytes())
                
                # Add audio spectrogram as a single feature with all values
                feature = example.feature_lists.feature_list['melspec/feature/floats'].feature.add()
                feature.float_list.value.extend(spectrogram.flatten())
                
                # Add label
                example.context.feature['label'].int64_list.value.append(label)
                
                # Write to TFRecord
                writer.write(example.SerializeToString())

if __name__ == '__main__':
    # Create sample TFRecords
    create_sample_tfrecords(
        output_dir='sample_tfrecords',
        num_examples=100,  # Total number of examples
        num_shards=5      # Number of shards
    )