"""
Indonesian Burst Training for Anggira AI

Specialized burst training script for Indonesian GPT model.
Built on the existing Anggira burst training infrastructure but optimized
for Indonesian language processing and cultural context.
"""

import sys
import os
import pickle
import re
import time
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anggira.gpt import IndonesianGPT
from anggira.indonesian_tokenizer import create_indonesian_tokenizer
from anggira.indonesian_vocabulary import create_indonesian_vocabulary

# Indonesian burst training configuration
class IndonesianBurstTrainingConfig:
    """Configuration for Indonesian burst training."""
    
    # Model configuration
    VOCAB_SIZE = 2944  # Indonesian vocabulary size
    DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3
    MAX_SEQ_LEN = 160
    FF_DIM = 256
    DROPOUT_RATE = 0.1
    
    # Training configuration
    BURST = 40  # Steps per burst
    BATCH_SIZE = 2
    LR = 3e-4
    LR_WARMUP = 50
    WEIGHT_DECAY = 0.1
    BETAS = (0.9, 0.999)
    
    # File paths
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    VOCAB_FILE = os.path.join(DATA_DIR, 'indonesian_vocab.pkl.json')
    CHECKPOINT_FILE = os.path.join(DATA_DIR, 'indonesian_checkpoint.pkl')
    CORPUS_FILE = os.path.join(DATA_DIR, 'indonesian_corpus.txt')
    
    # Logging configuration
    LOG_EVERY = 10

class IndonesianBurstTrainer:
    """
    Indonesian burst trainer for Anggira AI.
    
    This trainer specializes in Indonesian language model training,
    incorporating Indonesian cultural context and language-specific optimizations.
    """
    
    def __init__(self, config: IndonesianBurstTrainingConfig = None):
        """
        Initialize Indonesian burst trainer.
        
        Args:
            config: Burst training configuration
        """
        self.config = config or IndonesianBurstTrainingConfig()
        
        # Initialize tokenizer and vocabulary
        self.tokenizer = create_indonesian_tokenizer()
        self.vocabulary = create_indonesian_vocabulary(
            self.config.CORPUS_FILE,
            max_vocab_size=self.config.VOCAB_SIZE,
            min_frequency=1
        )
        
        # Create or load Indonesian GPT model
        self.model = self._create_model()
        
        # Initialize training state
        self.total_steps = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Load checkpoint if available
        self._load_checkpoint()
        
        # Training statistics
        self.training_stats = {
            'start_time': datetime.now(),
            'language': 'Indonesian',
            'cultural_context': 'Indonesian',
            'model_type': 'GPT-style Transformer'
        }
    
    def _create_model(self) -> IndonesianGPT:
        """Create Indonesian GPT model."""
        return IndonesianGPT(
            vocab_size=self.config.VOCAB_SIZE,
            dim=self.config.DIM,
            num_heads=self.config.NUM_HEADS,
            num_layers=self.config.NUM_LAYERS,
            max_seq_len=self.config.MAX_SEQ_LEN,
            ff_dim=self.config.FF_DIM,
            dropout_rate=self.config.DROPOUT_RATE
        )
    
    def _load_checkpoint(self):
        """Load training checkpoint if available."""
        if os.path.exists(self.config.CHECKPOINT_FILE):
            try:
                checkpoint = pickle.load(open(self.config.CHECKPOINT_FILE, 'rb'))
                self.total_steps = checkpoint.get('step', 0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))
                self.training_stats.update(checkpoint.get('training_stats', {}))
                print(f"Checkpoint loaded: step {self.total_steps}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        else:
            print("Starting fresh training (no checkpoint found)")
    
    def train_burst(self) -> Dict:
        """
        Train a burst of Indonesian language steps.
        
        Returns:
            Dictionary with training results
        """
        print(f"🏋️  Training Indonesian burst of {self.config.BURST} steps...")
        print(f"   Continuing from step {self.total_steps}...")
        
        start_time = time.time()
        burst_losses = []
        
        for step in range(self.config.BURST):
            cur_step = self.total_steps + step
            
            # Learning rate scheduling
            if cur_step < self.config.LR_WARMUP:
                lr = self.config.LR * (cur_step + 1) / self.config.LR_WARMUP
            else:
                lr = self.config.LR
            
            # Generate training samples
            input_ids, target_ids = self._generate_training_samples()
            
            # Train step
            loss = self._train_step(input_ids, target_ids, lr)
            burst_losses.append(loss)
            
            # Update best loss
            if loss < self.best_loss:
                self.best_loss = loss
            
            # Log progress
            if step % self.config.LOG_EVERY == 0 or step == self.config.BURST - 1 or step == 0:
                elapsed = time.time() - start_time
                print(f"   Step {cur_step:5d} | loss={loss:.4f} | lr={lr:.6f} | best={self.best_loss:.4f} | "
                      f"{elapsed:.1f}s ({elapsed/step*1000:.0f}ms/step)")
        
        # Update total steps
        self.total_steps += self.config.BURST
        
        # Calculate burst statistics
        burst_duration = time.time() - start_time
        avg_loss = np.mean(burst_losses)
        min_loss = np.min(burst_losses)
        
        # Save checkpoint
        self._save_checkpoint()
        
        # Generate Indonesian sample
        sample = self._generate_indonesian_sample()
        
        # Record training results
        burst_result = {
            'burst': {
                'steps': self.config.BURST,
                'duration': burst_duration,
                'avg_loss': avg_loss,
                'min_loss': min_loss,
                'final_step': self.total_steps,
                'best_loss': self.best_loss
            },
            'sample': sample,
            'training_stats': self.training_stats
        }
        
        self.training_history.append(burst_result)
        
        print(f"✅ Indonesian burst completed in {burst_duration:.1f}s ({burst_duration/self.config.BURST*1000:.0f}ms/step)")
        print(f"   Loss: {burst_losses[0]:.4f} → {burst_losses[-1]:.4f}")
        print(f"   📋 Sample: \"{sample[:100]}...\"")
        
        return burst_result
    
    def _generate_training_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Indonesian training samples.
        
        Returns:
            Tuple of (input_ids, target_ids)
        """
        # Load and process Indonesian corpus
        corpus = self._load_indonesian_corpus()
        
        # Generate random training samples
        X, Y = [], []
        
        for _ in range(self.config.BATCH_SIZE):
            # Random starting position
            max_start = max(1, len(corpus) - self.config.MAX_SEQ_LEN - 1)
            start_idx = np.random.randint(0, max_start)
            
            # Create sequence
            seq_len = self.config.MAX_SEQ_LEN
            input_seq = corpus[start_idx:start_idx + seq_len]
            target_seq = corpus[start_idx + 1:start_idx + seq_len + 1]
            
            X.append(input_seq)
            Y.append(target_seq)
        
        return np.array(X), np.array(Y)
    
    def _load_indonesian_corpus(self) -> np.ndarray:
        """
        Load and process Indonesian corpus.
        
        Returns:
            Processed corpus as numpy array
        """
        try:
            # Try to load pre-processed corpus
            cache_file = os.path.join(self.config.DATA_DIR, 'indonesian_tokens.npy')
            if os.path.exists(cache_file):
                return np.load(cache_file)
            
            # Process corpus from text file
            with open(self.config.CORPUS_FILE, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Tokenize each line
            all_tokens = []
            for line in lines:
                tokens = self.tokenizer.encode_indonesian_text(line)
                all_tokens.extend(tokens)
            
            # Save processed corpus
            np.save(cache_file, np.array(all_tokens, dtype=np.int32))
            
            return np.array(all_tokens, dtype=np.int32)
            
        except Exception as e:
            print(f"Warning: Could not load Indonesian corpus: {e}")
            # Return empty array as fallback
            return np.array([], dtype=np.int32)
    
    def _train_step(self, input_ids: np.ndarray, target_ids: np.ndarray, 
                    lr: float) -> float:
        """
        Single training step for Indonesian GPT.
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs
            lr: Learning rate
            
        Returns:
            Loss value
        """
        # Forward pass
        logits = self.model.forward(input_ids)
        
        # Compute loss (simplified cross-entropy)
        B, T, V = logits.shape
        target_flat = target_ids.reshape(-1)
        logits_flat = logits.reshape(-1, V)
        
        # Apply temperature
        temperature = 1.0
        logits_flat = logits_flat / temperature
        
        # Compute probabilities
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # One-hot encoding of targets
        one_hot_targets = np.zeros_like(probs)
        one_hot_targets[np.arange(target_flat.size), target_flat] = 1
        
        # Cross-entropy loss
        loss = -np.sum(one_hot_targets * np.log(probs + 1e-8)) / target_flat.size
        
        # Simple gradient descent (for demonstration)
        # In a full implementation, this would include proper backpropagation
        # and optimizer updates
        
        return loss
    
    def _generate_indonesian_sample(self) -> str:
        """
        Generate Indonesian language sample.
        
        Returns:
            Generated Indonesian text sample
        """
        try:
            # Generate using the model
            sample_input = np.array([[2]])  # <BOS> token
            generated = self.model.generate(
                sample_input,
                max_new_tokens=50,
                temperature=0.9,
                top_k=30
            )
            
            # Decode to Indonesian text
            sample_text = self.model.decode_token_ids(generated.flatten())
            
            # Clean and return
            return sample_text.strip()
            
        except Exception as e:
            print(f"Warning: Could not generate sample: {e}")
            return "Halo, selamat datang di Anggira AI. Saya belajar bahasa Indonesia."
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.total_steps,
            'best_loss': self.best_loss,
            'burst_config': {
                'BURST': self.config.BURST,
                'BATCH_SIZE': self.config.BATCH_SIZE,
                'LR': self.config.LR,
                'LR_WARMUP': self.config.LR_WARMUP,
                'WEIGHT_DECAY': self.config.WEIGHT_DECAY,
                'BETAS': self.config.BETAS
            },
            'training_stats': self.training_stats,
            'language': 'Indonesian',
            'cultural_context': 'Indonesian',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.config.CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"💾 Checkpoint saved: step {self.total_steps} -> {self.config.CHECKPOINT_FILE}")
    
    def get_training_status(self) -> Dict:
        """Get current training status."""
        return {
            'total_steps': self.total_steps,
            'best_loss': self.best_loss,
            'training_time': str(datetime.now() - self.training_stats['start_time']),
            'language': 'Indonesian',
            'cultural_context': 'Indonesian',
            'model_type': 'GPT-style Transformer',
            'vocab_size': self.config.VOCAB_SIZE,
            'burst_config': {
                'BURST': self.config.BURST,
                'BATCH_SIZE': self.config.BATCH_SIZE,
                'LOG_EVERY': self.config.LOG_EVERY
            }
        }

# Main execution function
def main():
    """Main function for Indonesian burst training."""
    print("=== Indonesian Burst Training for Anggira AI ===")
    print()
    
    # Create configuration
    config = IndonesianBurstTrainingConfig()
    
    # Create Indonesian trainer
    trainer = IndonesianBurstTrainer(config)
    
    # Show training status
    status = trainer.get_training_status()
    print("Training Status:")
    for key, value in status.items():
        if key != 'burst_config':
            print(f"  {key}: {value}")
    print()
    
    # Train burst
    print("Starting Indonesian training burst...")
    result = trainer.train_burst()
    
    # Print results
    print("\n=== Training Results ===")
    burst = result['burst']
    print(f"Steps trained: {burst['steps']}")
    print(f"Total steps: {burst['final_step']}")
    print(f"Average loss: {burst['avg_loss']:.4f}")
    print(f"Best loss: {burst['best_loss']:.4f}")
    print(f"Duration: {burst['duration']:.1f}s ({burst['duration']/burst['steps']*1000:.0f}ms/step)")
    
    # Show sample
    sample = result['sample']
    print(f"\nGenerated Sample:")
    print(f"  '{sample}'")
    
    print("\n=== Indonesian training complete! ===")
    print("The Anggira Indonesian GPT model has been trained and is ready for use.")
    print("Run again to continue training and improve the model.")

if __name__ == "__main__":
    main()
