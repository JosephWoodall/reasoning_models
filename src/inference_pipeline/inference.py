import sys
import os

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


from collections import Counter
import re
from typing import Dict, List
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import torch
from src.model.model import Transformer  # Import the main Transformer class


# Add the src directory to the Python path


class TextTokenizer:
    """Simple word-level tokenizer for text data."""

    def __init__(self, special_tokens: Dict[str, str]):
        self.special_tokens = special_tokens
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000):
        """Build vocabulary from text data."""
        # Add special tokens first
        for token in self.special_tokens.values():
            self.word_to_id[token] = len(self.word_to_id)
            self.id_to_word[len(self.id_to_word)] = token

        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            word_counts.update(words)

        # Add most frequent words to vocabulary
        most_common = word_counts.most_common(
            max_vocab_size - len(self.special_tokens))
        for word, _ in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
                self.id_to_word[len(self.id_to_word)] = word

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        unk_id = self.word_to_id[self.special_tokens['unk_token']]
        return [self.word_to_id.get(word, unk_id) for word in words]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        words = [self.id_to_word.get(token_id, self.special_tokens['unk_token'])
                 for token_id in token_ids]
        return ' '.join(words)


def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load the trained transformer model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
        model: Loaded model instance
        tokenizer: Tokenizer instance
    """
    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint loaded successfully")

    # Initialize tokenizer with special tokens (using the same as in training)
    special_tokens = {
        'pad_token': '<PAD>',
        'unk_token': '<UNK>',
        'bos_token': '<START>',
        'eos_token': '<END>'
    }
    tokenizer = TextTokenizer(special_tokens)

    # If checkpoint has word_to_id mapping, restore it
    if 'word_to_id' in checkpoint and 'id_to_word' in checkpoint:
        print("Found vocabulary in checkpoint")
        tokenizer.word_to_id = checkpoint['word_to_id']
        tokenizer.id_to_word = {
            int(k): v for k, v in checkpoint['id_to_word'].items()}
        tokenizer.vocab_size = len(tokenizer.word_to_id)
    else:
        raise ValueError("Checkpoint does not contain vocabulary information")

    # Get model configuration
    config = {}
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        if 'model' in checkpoint['config']:
            config = checkpoint['config']['model']
        else:
            config = checkpoint['config']

    # Use default values if config values are missing
    model_params = {
        'd_model': config.get('d_model', 128),
        'num_heads': config.get('num_heads', 4),
        'num_layers': config.get('num_layers', 2),
        'd_ff': config.get('d_ff', 512),
        'dropout': config.get('dropout', 0.1)
    }

    print(f"Creating model with params: {model_params}")

    # Create a new model instance with the same parameters used during training
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=model_params['d_model'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        d_ff=model_params['d_ff'],
        dropout=model_params['dropout']
    )

    print("Loading model state...")
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    print(f"Model loaded successfully and moved to {device}")

    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """
    Generate text using the trained model.

    Args:
        model: The loaded transformer model
        tokenizer: The tokenizer instance
        prompt (str): The starting text prompt
        max_length (int): Maximum length of generated sequence
        temperature (float): Controls randomness in generation (lower = more deterministic)

    Returns:
        str: Generated text
    """
    model.eval()  # Ensure model is in evaluation mode
    device = next(model.parameters()).device

    with torch.no_grad():  # No need to track gradients during inference
        # Convert prompt to tensor
        input_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_tokens]).to(
            device)  # Add batch dimension

        for _ in range(max_length):
            # Create target mask for the decoder
            tgt_mask = model.create_look_ahead_mask(
                input_ids.size(1)).to(device)

            # Get model predictions (we use the same sequence for both encoder and decoder)
            outputs = model(input_ids, input_ids, tgt_mask)
            next_token_logits = outputs[:, -1, :] / temperature

            # Sample from the output distribution
            next_token = torch.multinomial(
                F.softmax(next_token_logits, dim=-1), num_samples=1)

            # Append to input sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for end of sequence token
            if next_token.item() == tokenizer.word_to_id[tokenizer.special_tokens['eos_token']]:
                break

            # Safety check for length
            if input_ids.size(1) >= max_length:
                break

        # Convert back to text
        generated_text = tokenizer.decode(input_ids[0].tolist())

    return generated_text


if __name__ == "__main__":
    # Example usage
    checkpoint_path = Path("output/checkpoints/transformer_lm_latest.pt")

    # Force CPU usage due to CUDA compatibility issues
    device = 'cpu'
    torch.cuda.is_available = lambda: False  # Prevent CUDA usage

    try:
        # Load the model and tokenizer
        print("Loading model from checkpoint...")
        model, tokenizer = load_model(checkpoint_path, device)

        # Example prompt
        prompt = "Once upon a time"
        print(f"\nGenerating text from prompt: '{prompt}'")

        # Generate text
        generated_text = generate_text(
            model, tokenizer, prompt, max_length=100, temperature=0.7)
        print(f"\nGenerated text:\n{generated_text}")

    except Exception as e:
        print(f"Error: {str(e)}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
