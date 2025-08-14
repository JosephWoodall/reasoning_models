#!/usr/bin/env python3
"""
Transformer Model Training Script

This script trains a transformer model on text data from the text_cache directory.
It loads configuration from model_config.yml and implements a complete training pipeline.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import math
import random
import numpy as np
from collections import Counter
import re

# Add src directory to path to import model
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_dir)

# Now we can import the model
from model.model import Transformer

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextTokenizer:
    """Simple word-level tokenizer for text data."""

    def __init__(self, special_tokens: Dict[str, str]):
        self.special_tokens = special_tokens
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000):
        """Build vocabulary from text data."""
        print("Building vocabulary...")

        # Add special tokens first
        for token in self.special_tokens.values():
            self.word_to_id[token] = len(self.word_to_id)
            self.id_to_word[len(self.id_to_word)] = token

        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            # Simple tokenization: split on whitespace and punctuation
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
        print(f"Vocabulary size: {self.vocab_size}")

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


class TextDataset(Dataset):
    """Dataset class for text data."""

    def __init__(self, texts: List[str], tokenizer: TextTokenizer,
                 sequence_length: int, special_tokens: Dict[str, str]):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.special_tokens = special_tokens

        # Tokenize all texts and create sequences
        self.sequences = []
        pad_id = tokenizer.word_to_id[special_tokens['pad_token']]

        print("Creating training sequences...")
        for text in texts:
            token_ids = tokenizer.encode(text)

            # Create overlapping sequences
            for i in range(0, len(token_ids) - sequence_length + 1, sequence_length // 2):
                sequence = token_ids[i:i + sequence_length]
                if len(sequence) == sequence_length:
                    self.sequences.append(sequence)

        print(f"Created {len(self.sequences)} training sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # For language modeling: input is sequence[:-1], target is sequence[1:]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids


class WarmupLRScheduler:
    """Learning rate scheduler with warmup."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration. If YAML file exists, try to load it, otherwise use default config."""

    # Default configuration
    default_config = {
        'model': {
            'd_model': 128,     # Smaller model for testing
            'num_heads': 4,     # Fewer heads
            'num_layers': 2,    # Fewer layers
            'd_ff': 512,        # Smaller feed-forward
            'max_len': 1024,
            'dropout': 0.1,
            'src_vocab_size': 5000,  # Smaller vocab
            'tgt_vocab_size': 5000
        },
        'training': {
            'batch_size': 4,    # Much smaller batch for CPU
            'epochs': 5,        # Fewer epochs for testing
            'learning_rate': 0.001,  # Higher learning rate
            'scheduler': {
                'type': 'warmup',
                'warmup_steps': 100,  # Fewer warmup steps
                'step_factor': 0.5,
                'step_size': 10
            },
            'grad_clip': 1.0,
            'weight_decay': 0.01,
            'save_frequency': 2,    # Save more frequently
            'log_frequency': 10     # Log more frequently
        },
        'data': {
            'cache_dir': 'text_cache',
            'sequence_length': 128,  # Shorter sequences for faster training
            'min_text_length': 1000,
            'train_split': 0.9,
            'max_files': None,
            'special_tokens': {
                'pad_token': '<PAD>',
                'unk_token': '<UNK>',
                'start_token': '<START>',
                'end_token': '<END>'
            }
        },
        'output': {
            'checkpoint_dir': 'output/checkpoints',
            'log_dir': 'output/logs',
            'samples_dir': 'output/samples',
            'model_name': 'transformer_lm'
        },
        'generation': {
            'max_length': 200,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'num_samples': 5
        },
        'hardware': {
            'device': 'auto',  # Use CPU to avoid CUDA compatibility issues
            'num_workers': 2,   # Reduced for CPU
            'pin_memory': False  # No point with CPU
        },
        'evaluation': {
            'eval_frequency': 5,
            'metrics': ['perplexity', 'loss'],
            'eval_samples': 1000
        }
    }

    # Try to load YAML config if file exists
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            # Update default config with file config

            def update_config(default, update):
                for key, value in update.items():
                    if isinstance(value, dict) and key in default:
                        update_config(default[key], value)
                    else:
                        default[key] = value
            update_config(default_config, file_config)
        except ImportError:
            print("PyYAML not available, using default configuration")
        except Exception as e:
            print(
                f"Error loading config file: {e}, using default configuration")

    return default_config


def load_text_data(cache_dir: str, max_files: Optional[int] = None,
                   min_length: int = 1000) -> List[str]:
    """Load text data from cache directory."""
    print(f"Loading text data from {cache_dir}...")

    texts = []
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        raise ValueError(f"Cache directory {cache_dir} does not exist")

    text_files = list(cache_path.glob("*.txt"))
    if max_files:
        text_files = text_files[:max_files]

    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if len(text) >= min_length:
                    texts.append(text)
                    print(f"Loaded {file_path.name} ({len(text)} characters)")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(texts)} text files")
    return texts


def create_model(config: Dict, vocab_size: int) -> Transformer:
    """Create transformer model from configuration."""
    model_config = config['model']

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        d_ff=model_config['d_ff'],
        max_len=model_config['max_len'],
        dropout=model_config['dropout']
    )

    return model


def setup_directories(config: Dict):
    """Create necessary output directories."""
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    for key in ['checkpoint_dir', 'log_dir', 'samples_dir']:
        directory = os.path.join(project_root, config['output'][key])
        os.makedirs(directory, exist_ok=True)


def setup_logging(config: Dict):
    """Setup logging configuration."""
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    log_dir = os.path.join(project_root, config['output']['log_dir'])
    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int,
                    loss: float, config: Dict, tokenizer: TextTokenizer):
    """Save model checkpoint."""
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    checkpoint_dir = os.path.join(
        project_root, config['output']['checkpoint_dir'])
    model_name = config['output']['model_name']

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'vocab_size': tokenizer.vocab_size,
        'word_to_id': tokenizer.word_to_id,
        'id_to_word': tokenizer.id_to_word
    }

    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(
        checkpoint_dir, f'{model_name}_latest.pt'))

    # Save epoch-specific checkpoint
    torch.save(checkpoint, os.path.join(
        checkpoint_dir, f'{model_name}_epoch_{epoch}.pt'))

    logging.info(f"Saved checkpoint at epoch {epoch}")


def generate_sample(model: nn.Module, tokenizer: TextTokenizer,
                    device: torch.device, config: Dict, prompt: str = "") -> str:
    """Generate a text sample from the model."""
    model.eval()
    gen_config = config['generation']

    # Encode prompt
    if prompt:
        token_ids = tokenizer.encode(prompt)
    else:
        start_id = tokenizer.word_to_id[config['data']
                                        ['special_tokens']['start_token']]
        token_ids = [start_id]

    # Generate tokens
    with torch.no_grad():
        for _ in range(gen_config['max_length']):
            # Prepare input
            input_tensor = torch.tensor(
                [token_ids], dtype=torch.long).to(device)

            # Get model output
            output = model(input_tensor)
            logits = output[0, -1, :]  # Last token's logits

            # Apply temperature
            logits = logits / gen_config['temperature']

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            token_ids.append(int(next_token))

            # Check for end token
            if next_token == tokenizer.word_to_id[config['data']['special_tokens']['end_token']]:
                break

    return tokenizer.decode(token_ids)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer,
                scheduler, criterion, device: torch.device, config: Dict) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(input_ids)

        # Calculate loss
        loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

        # Update parameters
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

        # Log progress
        if batch_idx % config['training']['log_frequency'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')

    return total_loss / num_batches


def evaluate(model: nn.Module, dataloader: DataLoader, criterion,
             device: torch.device) -> float:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            output = model(input_ids)
            loss = criterion(output.view(-1, output.size(-1)),
                             target_ids.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    """Main training function."""
    # Load configuration from the same directory as this script
    config_path = os.path.join(os.path.dirname(__file__), 'model_config.yml')
    config = load_config(config_path)

    # Set random seed
    set_seed(42)

    # Setup directories and logging
    setup_directories(config)
    setup_logging(config)

    logging.info("Starting transformer training...")
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Determine device
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['hardware']['device'])

    logging.info(f"Using device: {device}")

    # Load text data (adjust path to be relative to project root)
    cache_dir = os.path.join(os.path.dirname(
        __file__), '..', '..', config['data']['cache_dir'])
    texts = load_text_data(
        cache_dir,
        config['data']['max_files'],
        config['data']['min_text_length']
    )

    if not texts:
        raise ValueError("No text data found!")

    # Split data
    split_idx = int(len(texts) * config['data']['train_split'])
    # Ensure we have at least one text for training
    if split_idx == 0:
        split_idx = 1
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:] if split_idx < len(
        texts) else [texts[-1]]  # Use last text for validation if no split

    logging.info(
        f"Train texts: {len(train_texts)}, Validation texts: {len(val_texts)}")

    # Create tokenizer and build vocabulary
    tokenizer = TextTokenizer(config['data']['special_tokens'])
    tokenizer.build_vocab(train_texts, config['model']['src_vocab_size'])

    # Create datasets
    train_dataset = TextDataset(
        train_texts, tokenizer, config['data']['sequence_length'],
        config['data']['special_tokens']
    )
    val_dataset = TextDataset(
        val_texts, tokenizer, config['data']['sequence_length'],
        config['data']['special_tokens']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    # Create model
    model = create_model(config, tokenizer.vocab_size)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logging.info(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Setup scheduler
    scheduler = None
    if config['training']['scheduler']['type'] == 'warmup':
        scheduler = WarmupLRScheduler(
            optimizer,
            config['model']['d_model'],
            config['training']['scheduler']['warmup_steps']
        )
    elif config['training']['scheduler']['type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['scheduler']['step_size'],
            gamma=config['training']['scheduler']['step_factor']
        )

    # Setup loss function
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.word_to_id[config['data']['special_tokens']['pad_token']])

    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    train_loss = 0.0  # Initialize train_loss

    for epoch in range(config['training']['epochs']):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, config)

        # Evaluate
        if epoch % config['evaluation']['eval_frequency'] == 0:
            val_loss = evaluate(model, val_loader, criterion, device)
            perplexity = math.exp(val_loss)

            logging.info(f'Epoch {epoch}/{config["training"]["epochs"]}:')
            logging.info(f'  Train Loss: {train_loss:.4f}')
            logging.info(f'  Val Loss: {val_loss:.4f}')
            logging.info(f'  Perplexity: {perplexity:.2f}')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler,
                                epoch, val_loss, config, tokenizer)

            # Generate sample
            sample = generate_sample(model, tokenizer, device, config, "The")
            logging.info(f'Sample: {sample[:200]}...')

        # Save periodic checkpoints
        if epoch % config['training']['save_frequency'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            train_loss, config, tokenizer)

        epoch_time = time.time() - epoch_start_time
        logging.info(f'Epoch {epoch} completed in {epoch_time:.2f}s')

    total_time = time.time() - start_time
    logging.info(f'Training completed in {total_time:.2f}s')

    # Final save
    save_checkpoint(model, optimizer, scheduler,
                    config['training']['epochs'], train_loss, config, tokenizer)

    logging.info("Training finished!")


if __name__ == "__main__":
    main()
