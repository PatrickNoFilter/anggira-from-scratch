"""
Anggira Indonesian Language Module

Indonesian language processing and training for Anggira AI.
Built from scratch following the same principles as other Anggira modules.

This module enables Anggira to understand, process, and generate
Indonesian language content for local and regional markets.

Author: Indonesian Language Learning Journey
"""

import re
import numpy as np
import pickle
import os
import json
from typing import List, Dict, Tuple, Any

class IndonesianTokenizer:
    """Indonesian text tokenizer for Anggira."""
    
    def __init__(self):
        # Indonesian word boundaries and patterns
        self.indonesian_word_pattern = r'[a-zA-Z0-9\']+(?:\'[a-zA-Z]+)?'
        self.special_tokens = {
            '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3
        }
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.current_idx = 4  # Start after special tokens
    
    def extract_indonesian_words(self, text: str) -> List[str]:
        """Extract Indonesian words from text using regex pattern."""
        # Clean the text
        text = text.lower().strip()
        # Find all Indonesian words using pattern
        words = re.findall(self.indonesian_word_pattern, text)
        return words
    
    def build_vocabulary(self, corpus_path: str, max_words: int = None):
        """Build Indonesian vocabulary from corpus file."""
        word_counts = {}
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = self.extract_indonesian_words(line)
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), 
                            key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary (cap at max_words if specified)
        for word, count in sorted_words:
            if max_words and len(self.word2idx) >= max_words:
                break
            if word not in self.word2idx:
                self.word2idx[word] = self.current_idx
                self.idx2word[self.current_idx] = word
                self.current_idx += 1
        
        return len(self.word2idx)
    
    def encode(self, text: str) -> List[int]:
        """Convert Indonesian text to token IDs."""
        words = self.extract_indonesian_words(text)
        token_ids = [self.word2idx.get(word, self.word2idx['<UNK>']) 
                    for word in words]
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to Indonesian text."""
        words = []
        for token_id in token_ids:
            if token_id in self.idx2word:
                word = self.idx2word[token_id]
                if word not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                    words.append(word)
        return ' '.join(words)

class IndonesianCorpusProcessor:
    """Process Indonesian corpus for training."""
    
    def __init__(self, tokenizer: IndonesianTokenizer):
        self.tokenizer = tokenizer
    
    def process_corpus(self, corpus_path: str, max_lines: int = None) -> np.ndarray:
        """Process Indonesian corpus into tokenized sequences."""
        all_tokens = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = list(f)
            if max_lines:
                lines = lines[:max_lines]
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Extract Indonesian words and add special tokens
                words = self.tokenizer.extract_indonesian_words(line)
                
                if words:  # Skip empty lines
                    # Add <BOS> token
                    token_ids = [self.tokenizer.word2idx['<BOS>']]
                    # Add word tokens
                    token_ids.extend([self.tokenizer.word2idx.get(word, 
                                     self.tokenizer.word2idx['<UNK>']) 
                                    for word in words])
                    # Add <EOS> token
                    token_ids.append(self.tokenizer.word2idx['<EOS>'])
                    
                    all_tokens.extend(token_ids)
        
        return np.array(all_tokens, dtype=np.int32)
    
    def create_sequences(self, tokens: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-target sequences for training."""
        X, Y = [], []
        
        for i in range(len(tokens) - seq_len):
            input_seq = tokens[i:i + seq_len]
            target_seq = tokens[i + 1:i + seq_len + 1]
            X.append(input_seq)
            Y.append(target_seq)
        
        return np.array(X), np.array(Y)

class IndonesianLanguageStats:
    """Statistics and analysis for Indonesian language processing."""
    
    @staticmethod
    def analyze_corpus(corpus_path: str) -> Dict[str, Any]:
        """Analyze Indonesian corpus statistics."""
        tokenizer = IndonesianTokenizer()
        corpus = IndonesianCorpusProcessor(tokenizer)
        
        # Read and analyze corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Calculate statistics
        total_words = 0
        total_lines = len(lines)
        unique_words = set()
        
        for line in lines:
            words = tokenizer.extract_indonesian_words(line)
            total_words += len(words)
            unique_words.update(words)
        
        return {
            'total_lines': total_lines,
            'total_words': total_words,
            'unique_words': len(unique_words),
            'vocabulary_size': len(tokenizer.word2idx),
            'average_words_per_line': total_words / max(total_lines, 1),
            'language': 'Indonesian',
            'corpus_type': 'Constitutional and Government Documents',
            'sample_content': lines[:3] if lines else []
        }

# Export public API
__all__ = [
    'IndonesianTokenizer',
    'IndonesianCorpusProcessor', 
    'IndonesianLanguageStats'
]