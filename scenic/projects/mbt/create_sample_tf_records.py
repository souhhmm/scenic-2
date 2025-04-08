import tensorflow as tf
import numpy as np
import cv2
import librosa
import os
import pandas as pd
from tqdm import tqdm
import warnings

def extract_frames(video_path, num_frames=32, stride=2):
    """Extract frames from video with proper resizing and normalization."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample
    if total_frames > num_frames * stride:
        # Sample frames with stride
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    else:
        # If video is shorter, sample all frames
        indices = np.arange(total_frames)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame to 256x256
        frame = cv2.resize(frame, (256, 256))
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    
    cap.release()
    return np.array(frames)

def extract_audio_spectrogram(video_path):
    """Extract mel spectrogram from the audio track of the video."""
    try:
        # Suppress librosa warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Load audio from video file
            audio, sr = librosa.load(video_path, sr=16000)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Reshape to match expected format (100, 128)
            mel_spec_db = librosa.util.fix_length(mel_spec_db, size=100, axis=0)
            mel_spec_db = librosa.util.fix_length(mel_spec_db, size=128, axis=1)
            
            return mel_spec_db
    except Exception as e:
        print(f"Warning: Could not extract audio from {video_path}: {str(e)}")
        # Return a very small value spectrogram instead of zeros
        # Using -80 dB (typical noise floor) as the minimum value
        return np.full((100, 128), -80.0, dtype=np.float32)

def create_multimodal_sequence_example(video_path, label):
    """Creates a SequenceExample with both video and audio data from the same video file."""
    example = tf.train.SequenceExample()
    
    # Context features (non-sequence data)
    example.context.feature['label'].int64_list.value.append(label)
    
    # RGB Frames
    frames = extract_frames(video_path)
    for frame in frames:
        frame_bytes = frame.tobytes()
        example.feature_lists.feature_list['rgb'].feature.add().bytes_list.value.append(frame_bytes)
    
    # Audio Spectrogram
    spec = extract_audio_spectrogram(video_path)
    spec_bytes = spec.tobytes()
    example.feature_lists.feature_list['melspec/feature/floats'].feature.add().bytes_list.value.append(spec_bytes)
    
    return example

def create_small_representative_dataset(df, samples_per_class=2):
    """Create a small representative dataset with samples from each class that have reliable audio."""
    # Filter for samples with reliable audio
    reliable_df = df[df['reliable_audio'] == 1]
    
    # Group by label and take specified number of samples from each class
    small_df = reliable_df.groupby('label').apply(lambda x: x.sample(min(samples_per_class, len(x)))).reset_index(drop=True)
    return small_df

def create_ssw60_tfrecords(output_dir, small_test=False, samples_per_class=2):
    """Create TFRecords for SSW60 dataset using video files that contain both video and audio.
    
    Args:
        output_dir: Directory to save TFRecords
        small_test: If True, creates a small representative dataset for testing
        samples_per_class: Number of samples to take from each class when small_test is True
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Direct Kaggle dataset paths
    video_dir = '/kaggle/input/birds60/ssw60/video_ml'
    video_csv = '/kaggle/input/birds60/ssw60/video_ml.csv'
    
    print(f"\nVideo directory: {video_dir}")
    
    # Load dataframe
    video_df = pd.read_csv(video_csv)
    
    print(f"\nVideo CSV shape: {video_df.shape}")
    print(f"Columns: {video_df.columns.tolist()}")
    
    # Process both train and test splits
    for split in ['train', 'test']:
        split_df = video_df[video_df['split'] == split]
        
        if small_test:
            split_df = create_small_representative_dataset(split_df, samples_per_class)
            print(f"\nCreated small representative {split} dataset with {samples_per_class} samples per class")
            print(f"Total samples in small {split} dataset: {len(split_df)}")
        
        output_path = os.path.join(output_dir, f'ssw60_multimodal_{split}.tfrecord')
        if small_test:
            output_path = os.path.join(output_dir, f'ssw60_multimodal_{split}_small.tfrecord')
        
        print(f"\nProcessing {split} split:")
        print(f"Number of examples: {len(split_df)}")
        
        with tf.io.TFRecordWriter(output_path) as writer:
            processed_count = 0
            skipped_count = 0
            no_audio_count = 0
            
            for _, row in tqdm(split_df.iterrows(), desc=f'Processing {split} split'):
                asset_id = row['asset_id']
                label = row['label']
                video_path = os.path.join(video_dir, f"{asset_id}.mp4")
                
                if not os.path.exists(video_path):
                    print(f"Warning: Video file not found: {video_path}")
                    skipped_count += 1
                    continue
                
                try:
                    example = create_multimodal_sequence_example(video_path, label)
                    writer.write(example.SerializeToString())
                    processed_count += 1
                    
                    # Check if audio was successfully extracted
                    cap = cv2.VideoCapture(video_path)
                    has_audio = cap.get(cv2.CAP_PROP_AUDIO_TOTAL_CHANNELS) > 0
                    cap.release()
                    if not has_audio:
                        no_audio_count += 1
                        
                except Exception as e:
                    print(f"Error processing video {asset_id}: {str(e)}")
                    skipped_count += 1
                    continue
            
            print(f"\n{split} split statistics:")
            print(f"Total examples: {len(split_df)}")
            print(f"Successfully processed: {processed_count}")
            print(f"Skipped (missing files): {skipped_count}")
            print(f"Videos with silent audio: {no_audio_count}")

if __name__ == '__main__':
    output_dir = 'tfrecords'  # Output directory for TFRecords
    # Create small test dataset with 2 samples per class
    create_ssw60_tfrecords(output_dir, small_test=True, samples_per_class=2)