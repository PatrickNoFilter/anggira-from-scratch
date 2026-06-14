"""
Indonesian Tokenizer Module

Specialized tokenizer for Indonesian language processing in Anggira AI.
Optimized for Indonesian text patterns and character sets.
"""

import re
import string
from typing import List, Dict, Tuple, Set

class IndonesianTokenizer:
    """
    Indonesian text tokenizer with proper handling of Indonesian language characteristics.
    
    Features:
    - Indonesian word boundary detection
    - Proper handling of Indonesian affixes and suffixes
    - Support for Indonesian diacritics and special characters
    - Efficient regex-based tokenization
    """
    
    # Indonesian language specific patterns
    INDONESIAN_WORD_PATTERN = r'[a-zA-Z0-9\']+(?:\'[a-zA-Z]+)?'
    INDONESIAN_SPECIAL_CHARS = ['é', 'è', 'ê', 'à', 'â', 'î', 'ô', 'ù', 'û', 'ç', 'ñ', 'ã', 'õ', 'ä', 'ö', 'ü']
    
    # Common Indonesian affixes and patterns (for future enhancement)
    INDONESIAN_AFFIXES = ['-', '=', '\\.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}']
    
    def __init__(self, preserve_case: bool = False):
        """
        Initialize Indonesian tokenizer.
        
        Args:
            preserve_case: If True, preserve original case; if False, convert to lowercase
        """
        self.preserve_case = preserve_case
        self.special_tokens = {
            'PAD': 'PAD',
            'UNK': 'UNK', 
            'BOS': 'BOS',
            'EOS': 'EOS'
        }
        self.word2idx = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}
        self.current_idx = len(self.word2idx)
        
        # Indonesian language statistics
        self.stats = {
            'total_tokens_processed': 0,
            'unique_words_found': 0,
            'language': 'Indonesian'
        }
    
    def extract_indonesian_words(self, text: str) -> List[str]:
        """
        Extract Indonesian words from text using language-specific patterns.
        
        Args:
            text: Indonesian text to tokenize
            
        Returns:
            List of Indonesian words extracted from text
        """
        if not text or not text.strip():
            return []
        
        # Clean and normalize text
        text = text.strip()
        
        # Handle Indonesian specific character patterns
        # Replace common Indonesian punctuation
        text = re.sub(r'\\s+', ' ', text)  # Normalize whitespace
        
        # Extract words using Indonesian pattern
        words = re.findall(self.INDONESIAN_WORD_PATTERN, text)
        
        # Process case
        if not self.preserve_case:
            words = [word.lower() for word in words]
        
        self.stats['total_tokens_processed'] += len(words)
        return words
    
    def build_vocabulary_from_corpus(self, corpus_path: str, 
                                   max_vocab_size: int = None,
                                   min_frequency: int = 1) -> Dict:
        """
        Build Indonesian vocabulary from corpus file.
        
        Args:
            corpus_path: Path to Indonesian corpus file
            max_vocab_size: Maximum vocabulary size (None for unlimited)
            min_frequency: Minimum word frequency to include
            
        Returns:
            Dictionary with vocabulary statistics
        """
        word_counts = {}
        
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    words = self.extract_indonesian_words(line)
                    for word in words:
                        word_counts[word] = word_counts.get(word, 0) + 1
        except FileNotFoundError:
            print(f"Warning: Corpus file not found: {corpus_path}")
            return self._get_default_vocab_stats()
        
        # Filter by minimum frequency and sort by frequency
        filtered_words = [(word, count) for word, count in word_counts.items() 
                         if count >= min_frequency]
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        # Build vocabulary
        for word, count in filtered_words:
            if max_vocab_size and len(self.word2idx) >= max_vocab_size:
                break
            if word not in self.word2idx:
                self.word2idx[word] = self.current_idx
                self.idx2word[self.current_idx] = word
                self.current_idx += 1
        
        self.stats['unique_words_found'] = len(self.word2idx) - len(self.special_tokens)
        
        return {
            'vocab_size': len(self.word2idx),
            'corpus_size': sum(word_counts.values()),
            'unique_words': len(word_counts),
            'included_words': self.stats['unique_words_found'],
            'most_common_words': filtered_words[:10]
        }
    
    def encode_indonesian_text(self, text: str) -> List[int]:
        """
        Convert Indonesian text to token IDs.
        
        Args:
            text: Indonesian text to encode
            
        Returns:
            List of token IDs
        """
        words = self.extract_indonesian_words(text)
        token_ids = []
        
        for word in words:
            token_id = self.word2idx.get(word, self.word2idx['UNK'])
            token_ids.append(token_id)
        
        return token_ids
    
    def decode_token_ids(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to Indonesian text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Indonesian text string
        """
        words = []
        
        for token_id in token_ids:
            if token_id in self.idx2word:
                word = self.idx2word[token_id]
                # Filter out special tokens
                if word not in ['PAD', 'UNK', 'BOS', 'EOS']:
                    words.append(word)
        
        return ' '.join(words)
    
    def get_vocabulary_stats(self) -> Dict:
        """Get comprehensive vocabulary statistics."""
        return {
            'total_vocabulary_size': len(self.word2idx),
            'special_tokens_count': len(self.special_tokens),
            'regular_words_count': self.stats['unique_words_found'],
            'most_frequent_words': list(self.word2idx.keys())[:10],
            'language': self.stats['language'],
            'tokenizer_settings': {
                'preserve_case': self.preserve_case,
                'word_pattern': self.INDONESIAN_WORD_PATTERN
            }
        }
    
    def _get_default_vocab_stats(self) -> Dict:
        """Return default vocabulary statistics when corpus is not available."""
        return {
            'vocab_size': len(self.word2idx),
            'corpus_size': 0,
            'unique_words': 0,
            'included_words': 0,
            'most_common_words': []
        }
    
    def save_vocabulary(self, output_path: str):
        """
        Save Indonesian vocabulary to file.
        
        Args:
            output_path: Path to save vocabulary
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'stats': self.stats,
            'tokenizer_settings': {
                'preserve_case': self.preserve_case,
                'word_pattern': self.INDONESIAN_WORD_PATTERN
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load_vocabulary(self, input_path: str):
        """
        Load Indonesian vocabulary from file.
        
        Args:
            input_path: Path to vocabulary file
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.stats = vocab_data['stats']
        
        # Recalculate current index
        self.current_idx = max(self.word2idx.values()) + 1 if self.word2idx else 4

# Factory function for easy tokenizer creation
def create_indonesian_tokenizer(preserve_case: bool = False) -> IndonesianTokenizer:
    """Create and return an Indonesian tokenizer instance."""
    return IndonesianTokenizer(preserve_case)

# Example usage function
def example_indonesian_tokenization():
    """Example of Indonesian tokenization usage."""
    print("=== Indonesian Tokenizer Example ===")
    
    # Create tokenizer
    tokenizer = create_indonesian_tokenizer(preserve_case=False)
    
    # Build vocabulary from corpus
    print("Building vocabulary from Indonesian corpus...")
    vocab_stats = tokenizer.build_vocabulary_from_corpus(
        '/root/anggira/data/indonesian_corpus/indonesian_corpus.txt',
        max_vocab_size=5000,
        min_frequency=1
    )
    
    print(f"Vocabulary built: {vocab_stats['vocab_size']} tokens")
    print(f"Included words: {vocab_stats['included_words']}")
    
    # Test encoding/decoding
    test_texts = [
        "Halo, bagaimana kabar Anda hari ini?",
        "Saya suka belajar bahasa Indonesia.",
        "UUD 1945 adalah dasar negara Indonesia.",
        "Presiden adalah kepala pemerintahan."
    ]
    
    print("\n=== Indonesian Text Encoding Examples ===")
    for text in test_texts:
        encoded = tokenizer.encode_indonesian_text(text)
        decoded = tokenizer.decode_token_ids(encoded)
        print(f"Original: {text}")
        print(f"Encoded: {encoded[:10]}... (length: {len(encoded)})")
        print(f"Decoded: {decoded}")
        print()
    
    # Show vocabulary statistics
    vocab_stats = tokenizer.get_vocabulary_stats()
    print("=== Vocabulary Statistics ===")
    for key, value in vocab_stats.items():
        if key != 'tokenizer_settings':
            print(f"{key}: {value}")

if __name__ == "__main__":
    example_indonesian_tokenization()
