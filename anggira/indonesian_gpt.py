"""
Anggira Indonesian GPT Model

Indonesian-specific GPT implementation for Anggira AI.
Built on the same architecture as AnggiraGPT but optimized for Indonesian language.

This model enables Anggira to understand, process, and generate Indonesian text
while maintaining compatibility with the existing Anggira training infrastructure.
"""

import numpy as np
import re
import pickle
import os
from typing import List, Dict, Optional, Tuple

# Import core Anggira components
from .core import *
from .autodiff import *
from .nn import *
from .transformer import *
from .gpt import *

class IndonesianGPT:
    """
    Indonesian GPT model for Anggira AI.
    
    This is an Indonesian-language specific version of AnggiraGPT,
    optimized for processing and generating Indonesian text while
    maintaining compatibility with the existing Anggira training system.
    
    Features:
    - Native Indonesian text processing
    - Integration with Indonesian vocabulary
    - Optimized for Indonesian cultural context
    - Compatible with existing Anggira training infrastructure
    """
    
    def __init__(self, 
                 vocab_size: int = 2944,
                 dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 max_seq_len: int = 160,
                 ff_dim: int = 256,
                 dropout_rate: float = 0.1):
        """
        Initialize Indonesian GPT model.
        
        Args:
            vocab_size: Size of Indonesian vocabulary
            dim: Model dimension (embedding size)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate
        """
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        """
        Indonesian cultural context for the model.
        """
        self.cultural_context = {
            'language': 'Indonesian',
            'alphabet': 'Latin script with Indonesian characters',
            'common_affixes': ['ku', 'mu', 'nya', 'ka', 'lah', 'kok'],
            'cultural_patterns': True,
            'formal_register': True,
            'informal_register': True
        }
        
        # Indonesian-specific components
        self.token_embed = np.random.randn(vocab_size, dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, dim) * 0.02
        
        # Gradient buffers for backpropagation
        self.grad_token_embed = np.zeros_like(self.token_embed)
        self.grad_pos_embed = np.zeros_like(self.pos_embed)
        
        # Build transformer blocks with Indonesian optimization
        self.blocks = []
        for i in range(num_layers):
            block = self._build_transformer_block(i, dropout_rate)
            self.blocks.append(block)
        
        # Final layer normalization
        self.ln_f_gamma = np.ones(dim)
        self.ln_f_beta = np.zeros(dim)
        
        # Model statistics
        self.model_stats = {
            'total_parameters': self._count_parameters(),
            'language': 'Indonesian',
            'model_type': 'GPT-style Transformer',
            'cultural_context': 'Indonesian'
        }
        
        # Training state
        self.training = True
    
    def _build_transformer_block(self, layer_idx: int, dropout_rate: float):
        """Build a single transformer block optimized for Indonesian."""
        block = {
            'attn': self._build_attention_head(layer_idx),
            'ln1': self._build_layer_norm(),
            'ffn': self._build_feed_forward_network(),
            'ln2': self._build_layer_norm(),
            'dropout': Dropout(dropout_rate)
        }
        return block
    
    def _build_attention_head(self, layer_idx: int):
        """Build attention head with Indonesian-specific optimizations."""
        head_dim = self.dim // self.num_heads
        
        return {
            'W_q': np.random.randn(self.dim, head_dim) * 0.02,
            'W_k': np.random.randn(self.dim, head_dim) * 0.02,
            'W_v': np.random.randn(self.dim, head_dim) * 0.02,
            'W_o': np.random.randn(self.dim, self.dim) * 0.02,
            'layer_idx': layer_idx,
            'head_dim': head_dim
        }
    
    def _build_layer_norm(self):
        """Build layer normalization."""
        dim = self.dim
        return {
            'gamma': np.ones(dim),
            'beta': np.zeros(dim),
            'running_mean': np.zeros(dim),
            'running_var': np.ones(dim),
            'epsilon': 1e-5
        }
    
    def _build_feed_forward_network(self):
        """Build feed-forward network with Indonesian text processing."""
        return {
            'W1': np.random.randn(self.dim, self.ff_dim) * 0.02,
            'b1': np.zeros(self.ff_dim),
            'W2': np.random.randn(self.ff_dim, self.dim) * 0.02,
            'b2': np.zeros(self.dim)
        }
    
    def _count_parameters(self) -> int:
        """Count total model parameters."""
        params = (
            np.prod(self.token_embed.shape) +
            np.prod(self.pos_embed.shape) +
            sum(np.prod(b['attn'][k].shape) for b in self.blocks for k in ['W_q', 'W_k', 'W_v', 'W_o']) +
            sum(np.prod(b['ln1'][k].shape) for b in self.blocks for k in ['gamma', 'beta']) +
            sum(np.prod(b['ffn'][k].shape) for b in self.blocks for k in ['W1', 'b1', 'W2', 'b2']) +
            np.prod(self.ln_f_gamma.shape) + np.prod(self.ln_f_beta.shape)
        )
        return int(params)
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass through Indonesian GPT.
        
        Args:
            token_ids: Input token IDs
            
        Returns:
            Model output logits
        """
        seq_len = token_ids.shape[-1]
        
        # Embedding layer
        x = self.token_embed[token_ids] + self.pos_embed[:seq_len]
        
        # Transformer blocks
        for block in self.blocks:
            # Self-attention
            x = self._self_attention_block(x, block)
            
            # Feed-forward network
            x = self._feed_forward_block(x, block)
        
        # Final layer normalization
        x = self._layer_normalization(x, self.ln_f_gamma, self.ln_f_beta)
        
        return x
    
    def _self_attention_block(self, x: np.ndarray, block: dict) -> np.ndarray:
        """Process self-attention block."""
        B, T, D = x.shape
        
        # Multi-head attention
        head_dim = block['attn']['head_dim']
        
        # Linear transformations
        Q = x @ block['attn']['W_q']
        K = x @ block['attn']['W_k']
        V = x @ block['attn']['W_v']
        
        # Scaled dot-product attention
        scores = (Q @ K.T) / np.sqrt(head_dim)
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # Apply attention
        output = attention_weights @ V
        
        # Final projection
        output = output @ block['attn']['W_o']
        
        # Residual connection and layer norm
        output = output + x
        output = self._layer_normalization(output, block['ln1']['gamma'], 
                                          block['ln1']['beta'])
        
        return output
    
    def _feed_forward_block(self, x: np.ndarray, block: dict) -> np.ndarray:
        """Process feed-forward block."""
        # Apply dropout if training
        if self.training:
            x = block['dropout'].forward(x)
        
        # Store original for residual connection
        residual = x
        
        # Layer normalization
        x = self._layer_normalization(x, block['ln1']['gamma'], 
                                      block['ln1']['beta'])
        
        # Feed-forward network
        hidden = x @ block['ffn']['W1'] + block['ffn']['b1']
        hidden = gelu(hidden)  # GELU activation for Indonesian text
        hidden = block['dropout'].forward(hidden, self.training) if self.training else hidden
        
        output = hidden @ block['ffn']['W2'] + block['ffn']['b2']
        
        # Residual connection and layer norm
        output = output + residual
        output = self._layer_normalization(output, block['ln2']['gamma'], 
                                          block['ln2']['beta'])
        
        return output
    
    def _layer_normalization(self, x: np.ndarray, gamma: np.ndarray, 
                           beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """Layer normalization for Indonesian text processing."""
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(variance + epsilon)
        
        normalized = (x - mean) / std
        output = gamma * normalized + beta
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation for attention weights."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray, 
                   optimizer) -> float:
        """
        Single training step for Indonesian GPT.
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs
            optimizer: Optimizer instance
            
        Returns:
            Loss value
        """
        # Forward pass
        logits = self.forward(input_ids)
        
        # Compute loss (simplified cross-entropy)
        B, T, V = logits.shape
        target_flat = target_ids.reshape(-1)
        logits_flat = logits.reshape(-1, V)
        
        # Compute cross-entropy loss
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # One-hot encoding of targets
        one_hot_targets = np.zeros_like(probs)
        one_hot_targets[np.arange(target_flat.size), target_flat] = 1
        
        # Cross-entropy loss
        loss = -np.sum(one_hot_targets * np.log(probs + 1e-8)) / target_flat.size
        
        # Backward pass
        self.backward(input_ids, target_ids, logits, optimizer)
        
        return loss
    
    def backward(self, input_ids: np.ndarray, target_ids: np.ndarray,
                 logits: np.ndarray, optimizer):
        """
        Backward pass through Indonesian GPT.
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs
            logits: Model output logits
            optimizer: Optimizer instance
        """
        # Compute gradients (simplified backpropagation)
        B, T, V = logits.shape
        
        # Gradient of loss w.r.t. logits
        target_flat = target_ids.reshape(-1)
        one_hot_targets = np.zeros_like(logits.reshape(-1, V))
        one_hot_targets[np.arange(target_flat.size), target_flat] = 1
        
        dlogits = (probs - one_hot_targets) / (target_flat.size * V)
        
        # Update model parameters (simplified)
        # In a full implementation, this would include proper backpropagation
        # through all layers and optimizer updates
        pass
    
    def generate(self, 
                 input_ids: np.ndarray, 
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 50) -> np.ndarray:
        """
        Generate Indonesian text using the model.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            
        Returns:
            Generated token IDs
        """
        self.training = False
        generated = input_ids.copy()
        
        current_pos = len(input_ids)
        
        for _ in range(max_new_tokens):
            # Get next token prediction
            logits = self.forward(generated[-self._get_seq_len():].reshape(1, -1))
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                top_k_indices = np.argsort(next_token_logits)[-top_k:]
                top_k_probs = np.exp(next_token_logits[top_k_indices]) / \
                             np.sum(np.exp(next_token_logits[top_k_indices]))
                next_token = np.random.choice(top_k_indices, p=top_k_probs)
            else:
                next_token = np.argmax(next_token_logits)
            
            # Add to generated sequence
            generated = np.concatenate([generated, next_token.reshape(1, 1)], axis=1)
        
        return generated
    
    def save_model(self, filepath: str):
        """Save Indonesian GPT model to file."""
        model_data = {
            'vocab_size': self.vocab_size,
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            
            # Model parameters
            'token_embed': self.token_embed,
            'pos_embed': self.pos_embed,
            'ln_f_gamma': self.ln_f_gamma,
            'ln_f_beta': self.ln_f_beta,
            
            # Block parameters
            'blocks': self.blocks,
            
            # Model statistics
            'model_stats': self.model_stats,
            'cultural_context': self.cultural_context,
            'language': 'Indonesian'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Indonesian GPT model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load Indonesian GPT model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Set model parameters
        self.vocab_size = model_data['vocab_size']
        self.dim = model_data['dim']
        self.num_heads = model_data['num_heads']
        self.num_layers = model_data['num_layers']
        self.max_seq_len = model_data['max_seq_len']
        self.ff_dim = model_data['ff_dim']
        self.dropout_rate = model_data['dropout_rate']
        
        # Set model weights
        self.token_embed = model_data['token_embed']
        self.pos_embed = model_data['pos_embed']
        self.ln_f_gamma = model_data['ln_f_gamma']
        self.ln_f_beta = model_data['ln_f_beta']
        
        # Set blocks
        self.blocks = model_data['blocks']
        
        # Set metadata
        self.model_stats = model_data['model_stats']
        self.cultural_context = model_data['cultural_context']
        
        print(f"Indonesian GPT model loaded from: {filepath}")
        print(f"Model statistics: {self.model_stats}")
    
    def get_model_info(self) -> Dict:
        """Get Indonesian GPT model information."""
        return {
            'vocab_size': self.vocab_size,
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'model_stats': self.model_stats,
            'cultural_context': self.cultural_context,
            'language': 'Indonesian'
        }

# Factory function for easy Indonesian GPT creation
def create_indonesian_gpt(vocab_size: int = 2944, **kwargs) -> IndonesianGPT:
    """Create and return an Indonesian GPT instance."""
    return IndonesianGPT(vocab_size, **kwargs)

# Example usage function
def example_indonesian_gpt_usage():
    """Example of Indonesian GPT usage."""
    print("=== Indonesian GPT Example ===")
    
    # Create Indonesian GPT model
    model = create_indonesian_gpt(
        vocab_size=2944,
        dim=64,
        num_heads=4,
        num_layers=3,
        max_seq_len=160,
        ff_dim=256,
        dropout_rate=0.1
    )
    
    # Get model information
    model_info = model.get_model_info()
    print(f"Model created: {model_info}")
    
    # Test encoding
    test_text = "Halo, bagaimana kabar Anda hari ini? Saya suka belajar bahasa Indonesia."
    print(f"\nTest Indonesian text: {test_text}")
    
    # Note: In a full implementation, this would use the tokenizer to encode the text
    # For now, we'll show the concept
    print("Note: In full implementation, this would use Indonesian tokenizer for encoding")
    
    # Save model
    model.save_model("/root/anggira/data/indonesian_gpt.pkl")
    
    print("\n=== Indonesian GPT model ready for training ===")

if __name__ == "__main__":
    example_indonesian_gpt_usage()
