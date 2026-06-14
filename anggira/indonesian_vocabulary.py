"""
Indonesian Vocabulary Builder

Builds and manages Indonesian vocabulary for Anggira AI.
Creates word2idx and idx2word mappings from Indonesian corpus.
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class IndonesianVocabularyBuilder:
    """
    Builds Indonesian vocabulary from corpus data.
    
    Features:
    - Creates word frequency analysis
    - Builds word2idx and idx2word mappings
    - Handles special tokens and unknown words
    - Supports vocabulary saving and loading
    - Provides vocabulary statistics
    """
    
    def __init__(self, 
                 max_vocab_size: int = 8000,
                 min_frequency: int = 1,
                 add_special_tokens: bool = True):
        """
        Initialize Indonesian vocabulary builder.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_frequency: Minimum word frequency to include
            add_special_tokens: Whether to add special tokens (PAD, UNK, BOS, EOS)
        """
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
        self.add_special_tokens = add_special_tokens
        
        # Vocabulary mappings
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_frequency: Dict[str, int] = {}
        
        # Special tokens
        self.special_tokens = {
            'PAD': 'pad_token',
            'UNK': 'unknown_token',
            'BOS': 'beginning_of_sequence',
            'EOS': 'end_of_sequence'
        }
        
        # Statistics
        self.stats = {
            'corpus_size': 0,
            'unique_words': 0,
            'vocab_size': 0,
            'language': 'Indonesian',
            'corpus_source': None,
            'build_date': None
        }
    
    def build_vocabulary_from_corpus(self, 
                                   corpus_path: str,
                                   tokenizer_module) -> Dict:
        """
        Build Indonesian vocabulary from corpus file using tokenizer.
        
        Args:
            corpus_path: Path to Indonesian corpus file
            tokenizer_module: Tokenizer module with extract_indonesian_words method
            
        Returns:
            Dictionary with vocabulary statistics
        """
        print(f"Building Indonesian vocabulary from: {corpus_path}")
        
        # Count word frequencies
        word_counts = {}
        total_words = 0
        total_lines = 0
        
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    total_lines += 1
                    # Extract Indonesian words using tokenizer
                    words = tokenizer_module.extract_indonesian_words(line)
                    total_words += len(words)
                    
                    for word in words:
                        word_counts[word] = word_counts.get(word, 0) + 1
            
            # Update statistics
            self.stats['corpus_size'] = total_words
            self.stats['total_lines'] = total_lines
            
        except FileNotFoundError:
            print(f"Error: Corpus file not found: {corpus_path}")
            return self._get_default_stats()
        
        # Add special tokens first
        if self.add_special_tokens:
            for token_name, token_value in self.special_tokens.items():
                self.word2idx[token_value] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = token_value
        
        # Filter by minimum frequency and sort by frequency
        filtered_words = [(word, count) for word, count in word_counts.items() 
                         if count >= self.min_frequency]
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        # Build vocabulary (respect max size)
        words_added = 0
        for word, frequency in filtered_words:
            if self.max_vocab_size and len(self.word2idx) >= self.max_vocab_size:
                break
            
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word
                self.word_frequency[word] = frequency
                words_added += 1
        
        # Update final statistics
        self.stats['unique_words'] = len(word_counts)
        self.stats['vocab_size'] = len(self.word2idx)
        self.stats['words_added'] = words_added
        self.stats['corpus_source'] = corpus_path
        self.stats['build_date'] = str(np.datetime64('now'))
        
        return self._get_comprehensive_stats()
    
    def create_tokenizer_compatible_mappings(self) -> Tuple[Dict, Dict]:
        """
        Create word2idx and idx2word mappings compatible with Indonesian tokenizer.
        
        Returns:
            Tuple of (word2idx, idx2word) dictionaries
        """
        # Create mappings compatible with IndonesianTokenizer
        tokenizer_word2idx = {word: idx for idx, word in self.word2idx.items()}
        tokenizer_idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        return tokenizer_word2idx, tokenizer_idx2word
    
    def save_vocabulary(self, output_path: str):
        """
        Save Indonesian vocabulary to files.
        
        Args:
            output_path: Base path for saving vocabulary files
        """
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON (human-readable)
        json_path = f"{output_path}.json"
        json_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_frequency': self.word_frequency,
            'stats': self.stats,
            'tokenizer_settings': {
                'max_vocab_size': self.max_vocab_size,
                'min_frequency': self.min_frequency,
                'add_special_tokens': self.add_special_tokens
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Save as pickle (for Python compatibility)
        pickle_path = f"{output_path}.pkl"
        pickle.dump({
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_frequency': self.word_frequency,
            'stats': self.stats
        }, open(pickle_path, 'wb'))
        
        print(f"Vocabulary saved to: {json_path} and {pickle_path}")
        return json_path, pickle_path
    
    def load_vocabulary(self, input_path: str):
        """
        Load Indonesian vocabulary from file.
        
        Args:
            input_path: Path to vocabulary file (with .json or .pkl extension)
        """
        if input_path.endswith('.json'):
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif input_path.endswith('.pkl'):
            data = pickle.load(open(input_path, 'rb'))
        else:
            raise ValueError("Unsupported file format. Use .json or .pkl")
        
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.word_frequency = data.get('word_frequency', {})
        self.stats = data.get('stats', self.stats)
        
        print(f"Vocabulary loaded from: {input_path}")
    
    def get_vocabulary_info(self) -> Dict:
        """Get comprehensive vocabulary information."""
        return {
            'vocab_size': len(self.word2idx),
            'special_tokens_count': len([w for w in self.word2idx.keys() 
                                        if w in self.special_tokens.values()]),
            'regular_words_count': len([w for w in self.word2idx.keys() 
                                      if w not in self.special_tokens.values()]),
            'max_frequency': max(self.word_frequency.values()) if self.word_frequency else 0,
            'min_frequency': min(self.word_frequency.values()) if self.word_frequency else 0,
            'average_frequency': (sum(self.word_frequency.values()) / 
                                len(self.word_frequency) if self.word_frequency else 0),
            'language': self.stats['language'],
            'settings': {
                'max_vocab_size': self.max_vocab_size,
                'min_frequency': self.min_frequency,
                'add_special_tokens': self.add_special_tokens
            }
        }
    
    def encode_indonesian_text(self, text: str) -> List[int]:
        """
        Encode Indonesian text to token IDs.
        
        Args:
            text: Indonesian text to encode
            
        Returns:
            List of token IDs
        """
        words = text.lower().split()
        token_ids = []
        
        for word in words:
            token_id = self.word2idx.get(word, self.word2idx.get('UNK', 1))
            token_ids.append(token_id)
        
        return token_ids
    
    def decode_token_ids(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to Indonesian text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Indonesian text
        """
        words = []
        
        for token_id in token_ids:
            if token_id in self.idx2word:
                word = self.idx2word[token_id]
                # Filter out special tokens
                if word not in ['pad_token', 'unknown_token', 'beginning_of_sequence', 'end_of_sequence']:
                    words.append(word)
        
        return ' '.join(words)
    
    def get_most_frequent_words(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get most frequent Indonesian words.
        
        Args:
            n: Number of most frequent words to return
            
        Returns:
            List of (word, frequency) tuples
        """
        sorted_words = sorted(self.word_frequency.items(), 
                            key=lambda x: x[1], reverse=True)
        return sorted_words[:n]
    
    def _get_default_stats(self) -> Dict:
        """Get default statistics when corpus is not available."""
        return {
            'vocab_size': len(self.word2idx),
            'corpus_size': 0,
            'unique_words': 0,
            'words_added': 0,
            'most_common_words': []
        }
    
    def _get_comprehensive_stats(self) -> Dict:
        """Get comprehensive vocabulary statistics."""
        return {
            'vocab_size': len(self.word2idx),
            'corpus_size': self.stats['corpus_size'],
            'unique_words': self.stats['unique_words'],
            'words_added': self.stats.get('words_added', 0),
            'most_common_words': self.get_most_frequent_words(10)
        }

# Factory function for easy vocabulary creation
def create_indonesian_vocabulary(corpus_path: str,
                                max_vocab_size: int = 8000,
                                min_frequency: int = 1) -> IndonesianVocabularyBuilder:
    """
    Create and return an Indonesian vocabulary builder instance.
    
    Args:
        corpus_path: Path to Indonesian corpus file
        max_vocab_size: Maximum vocabulary size
        min_frequency: Minimum word frequency
        
    Returns:
        IndonesianVocabularyBuilder instance
    """
    from .indonesian_tokenizer import create_indonesian_tokenizer
    
    tokenizer = create_indonesian_tokenizer()
    vocab_builder = IndonesianVocabularyBuilder(
        max_vocab_size=max_vocab_size,
        min_frequency=min_frequency
    )
    
    vocab_builder.build_vocabulary_from_corpus(corpus_path, tokenizer)
    return vocab_builder

# Example usage function
def example_indonesian_vocabulary_building():
    """Example of Indonesian vocabulary building usage."""
    print("=== Indonesian Vocabulary Builder Example ===")
    
    # Create vocabulary builder
    vocab_builder = create_indonesian_vocabulary(
        '/root/anggira/data/indonesian_corpus/indonesian_corpus.txt',
        max_vocab_size=5000,
        min_frequency=1
    )
    
    # Get vocabulary information
    vocab_info = vocab_builder.get_vocabulary_info()
    print(f"Vocabulary built: {vocab_info['vocab_size']} tokens")
    print(f"Regular words: {vocab_info['regular_words_count']}")
    print(f"Special tokens: {vocab_info['special_tokens_count']}")
    
    # Test encoding/decoding
    test_texts = [
        "Halo, bagaimana kabar Anda hari ini?",
        "Saya suka belajar bahasa Indonesia.",
        "UUD 1945 adalah dasar negara Indonesia.",
        "Presiden adalah kepala pemerintahan."
    ]
    
    print("\n=== Indonesian Text Encoding Examples ===")
    for text in test_texts:
        encoded = vocab_builder.encode_indonesian_text(text)
        decoded = vocab_builder.decode_token_ids(encoded)
        print(f"Original: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print()
    
    # Save vocabulary
    vocab_path = "/root/anggira/data/indonesian_vocab"
    vocab_builder.save_vocabulary(vocab_path)
    
    # Show most frequent words
    most_frequent = vocab_builder.get_most_frequent_words(5)
    print("=== Most Frequent Indonesian Words ===")
    for word, freq in most_frequent:
        print(f"{word}: {freq}")

if __name__ == "__main__":
    example_indonesian_vocabulary_building()
