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
from src.model.model import Transformer  

class TextTokenizer:
    """Simple word-level tokenizer for text data."""

    def __init__(self, special_tokens: Dict[str, str]):
        self.special_tokens = special_tokens
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000):
        """Build vocabulary from text data."""
        for token in self.special_tokens.values():
            self.word_to_id[token] = len(self.word_to_id)
            self.id_to_word[len(self.id_to_word)] = token

        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            word_counts.update(words)

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


def load_model(checkpoint_path, device='cpu'):
    """
    Load the trained transformer model and BPE tokenizer.

    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
        model: Loaded model instance
        tokenizer: Tokenizer instance
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=torch.serialization.pickle)
    except:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint loaded successfully")

    # Load the BPE tokenizer from the JSON file
    tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Loaded BPE tokenizer from {tokenizer_path}")
    
    class TokenizerWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.vocab_size = tokenizer.get_vocab_size()
            
        def encode(self, text: str) -> List[int]:
            return self.tokenizer.encode(text).ids
            
        def decode(self, token_ids: List[int]) -> str:
            return self.tokenizer.decode(token_ids)
    
    wrapped_tokenizer = TokenizerWrapper(tokenizer)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    vocab_size, d_model = state_dict['src_embedding.weight'].shape
    d_ff = state_dict['encoder_layers.0.feed_forward.linear1.weight'].size(0)
    num_layers = sum(1 for key in state_dict if key.startswith('encoder_layers.') 
                    and key.endswith('.feed_forward.linear1.weight'))
    
    print(f"\nModel architecture:")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model dimension: {d_model}")
    print(f"Feed-forward dimension: {d_ff}")
    print(f"Number of layers: {num_layers}")
    
    model_args = {
        'src_vocab_size': vocab_size,
        'tgt_vocab_size': vocab_size,
        'd_model': d_model,
        'num_heads': 8,  # This could be extracted if needed
        'num_layers': num_layers,
        'd_ff': d_ff,
        'max_len': 5000,
        'dropout': 0.1
    }
    
    model = Transformer(**model_args)
    try:
        model.load_state_dict(state_dict)
        print("\nModel state loaded successfully")
    except Exception as e:
        print(f"\nWarning: Error loading state dict: {str(e)}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    return model, wrapped_tokenizer
            


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """
    Generate text using the trained model with nucleus sampling.

    Args:
        model: The loaded transformer model
        tokenizer: The tokenizer instance (TokenizerWrapper)
        prompt (str): The starting text prompt
        max_length (int): Maximum length of generated sequence
        temperature (float): Controls randomness (lower = more deterministic)
        top_p (float): Nucleus sampling parameter (0.9 means top 90% of probability mass)

    Returns:
        str: Generated text
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        generated_ids = []
        
        for _ in range(max_length):
            outputs = model(input_ids)  
            next_token_logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                probs[sorted_indices[sorted_indices_to_remove]] = 0
                probs = probs / probs.sum()
            
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token.item())
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.tokenizer.token_to_id("</s>"):
                break
                
            if len(generated_ids) >= max_length:
                break
        
        return tokenizer.decode(generated_ids)

    return generated_text


if __name__ == "__main__":
    checkpoint_path = Path("/home/joseph_woodall/workspace/reasoning_models/output/checkpoints/transformer_lm_latest.pt")

    device = 'cuda'

    try:
        print("="*50)
        print("Loading model from checkpoint...")
        model, tokenizer = load_model(checkpoint_path, device)
        print(model)
        print(tokenizer)
        print("="*50)
        
        
        print("="*50)
        print("Please enter a prompt:")
        prompt = input()
        print(f"\nGenerating text from prompt: '{prompt}'")
        print("="*50)
        generated_text = generate_text(
            model, tokenizer, prompt, max_length=100, temperature=0.7)
        print("\n")
        print("="*50)
        print(f"\nGenerated text:\n{generated_text}")
        print("="*50)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
