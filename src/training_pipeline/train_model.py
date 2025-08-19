#!/usr/bin/env python3
"""
Transformer Model Training Script (BPE tokenizer edition)

Key changes vs your original:
- Replaces custom word-level tokenizer with a trained BPE tokenizer (Hugging Face `tokenizers`).
- Removes <UNK>-heavy behavior; no padding is used since sequences are fixed-length.
- Saves/loads the tokenizer JSON next to checkpoints.
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
import numpy
import re

# Add src directory to path to import model
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_dir)

# Now we can import the model
from model.model import Transformer

# ------------- Utils -------------
def set_seed(seed: int = 42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------- BPE Tokenizer -------------
# We keep this self-contained in this file so you can drop it in and run
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import BertProcessing

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<START>", "<END>"]

def train_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 32000,
    save_path: str = "tokenizer.json",
) -> Tokenizer:
    """
    Train a Byte-Pair Encoding tokenizer from raw texts and save it to JSON.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    # Use ByteLevel pre-tokenizer for better subword tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        continuing_subword_prefix="##"
    )

    # Train directly from the iterator of texts (no temp files needed)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Post-processor: wrap sequences with <START> ... <END>
    # (Not strictly required for LM, but useful for generation)
    start_id = tokenizer.token_to_id("<START>")
    end_id = tokenizer.token_to_id("<END>")
    tokenizer.post_processor = BertProcessing(
        ("<END>", end_id),
        ("<START>", start_id),
    )

    tokenizer.save(save_path)
    return tokenizer

# ------------- Dataset -------------
class TextDataset(Dataset):
    """
    Dataset that:
      - Encodes each raw text with the BPE tokenizer
      - Slices into fixed-length sequences with 50% overlap (no padding)
      - Returns (input_ids, target_ids) shifted by one for LM
    """

    def __init__(self, texts: List[str], tokenizer: Tokenizer, sequence_length: int):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.sequences = []

        print("Encoding texts and creating sequences...")
        for text in texts:
            # encode returns an Encoding; .ids are token ids
            token_ids = tokenizer.encode(text).ids

            # Optional: ensure we always see start/end wrappers at text level
            # (The post-processor does this only when calling tokenizer.encode,
            #  which we already do above.)
            # Now slice into overlapping fixed-length chunks
            step = max(1, sequence_length // 2)
            for i in range(0, len(token_ids) - sequence_length, step):
                seq = token_ids[i:i + sequence_length]
                if len(seq) == sequence_length:
                    self.sequences.append(seq)

        print(f"Created {len(self.sequences)} training sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids

# ------------- Scheduler -------------
class WarmupLRScheduler:
    """Learning rate scheduler with warmup (Transformer-style)."""
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

# ------------- Config / IO -------------
def load_config(config_path: Optional[str] = None) -> Dict:
    """Load YAML config; falls back to default if not found."""
    default_config = {
        'model': {
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'd_ff': 512,
            'max_len': 1024,
            'dropout': 0.1,
            'src_vocab_size': 32000,
            'tgt_vocab_size': 32000
        },
        'training': {
            'batch_size': 4,
            'epochs': 5,
            'learning_rate': 0.001,
            'scheduler': {
                'type': 'warmup',
                'warmup_steps': 100,
                'step_factor': 0.5,
                'step_size': 10
            },
            'grad_clip': 1.0,
            'weight_decay': 0.01,
            'save_frequency': 2,
            'log_frequency': 10
        },
        'data': {
            'cache_dir': 'text_cache',
            'sequence_length': 128,
            'min_text_length': 1000,
            'train_split': 0.9,
            'max_files': None,
            'special_tokens': {  # kept for completeness; BPE uses SPECIAL_TOKENS above
                'pad_token': '<PAD>',
                'unk_token': '<UNK>',
                'start_token': '<START>',
                'end_token': '<END>'
            },
            'load_batch_size': 1000
        },
        'output': {
            'checkpoint_dir': 'output/checkpoints',
            'log_dir': 'output/logs',
            'samples_dir': 'output/samples',
            'model_name': 'transformer_lm',
            'tokenizer_path': 'output/checkpoints/tokenizer.json'
        },
        'generation': {
            'max_length': 200,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'num_samples': 5
        },
        'hardware': {
            'device': 'auto',
            'num_workers': 2,
            'pin_memory': False
        },
        'evaluation': {
            'eval_frequency': 5,
            'metrics': ['perplexity', 'loss'],
            'eval_samples': 1000
        }
    }

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)

            def update_config(default, update):
                for key, value in update.items():
                    if isinstance(value, dict) and key in default:
                        update_config(default[key], value)
                    else:
                        default[key] = value

            update_config(default_config, file_config)
        except Exception as e:
            print(f"Error loading config file: {e}, using default configuration")

    return default_config

def load_text_data(cache_dir: str, max_files: Optional[int] = None,
                   min_length: int = 1000, batch_size: int = 1000) -> List[str]:
    logging.info(f"Loading text data from {cache_dir}...")
    texts = []
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise ValueError(f"Cache directory {cache_dir} does not exist")

    text_files = list(cache_path.glob("*.txt"))
    random.shuffle(text_files)
    if max_files:
        text_files = text_files[:max_files]

    total_files = len(text_files)
    processed_files = 0
    total_chars = 0

    for i in range(0, len(text_files), batch_size):
        batch_files = text_files[i:i + batch_size]
        batch_texts = []
        for file_path in batch_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if len(text) >= min_length:
                        batch_texts.append(text)
                        total_chars += len(text)
                processed_files += 1
                if processed_files % 1000 == 0 or processed_files == total_files:
                    logging.info(
                        f"Progress: {processed_files}/{total_files} files "
                        f"({(processed_files/total_files)*100:.1f}%)"
                    )
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
        texts.extend(batch_texts)
        logging.info(
            f"Loaded batch of {len(batch_texts)} texts "
            f"(Total: {len(texts)} texts, {total_chars/1_000_000:.1f}M chars)"
        )

    logging.info(
        f"Completed loading {len(texts)} text files "
        f"({total_chars/1_000_000:.1f}M characters)"
    )
    return texts

def create_model(config: Dict, vocab_size: int) -> Transformer:
    m = config['model']
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=m['d_model'],
        num_heads=m['num_heads'],
        num_layers=m['num_layers'],
        d_ff=m['d_ff'],
        max_len=m['max_len'],
        dropout=m['dropout']
    )
    return model

def setup_directories(config: Dict):
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    for key in ['checkpoint_dir', 'log_dir', 'samples_dir']:
        directory = os.path.join(project_root, config['output'][key])
        os.makedirs(directory, exist_ok=True)

def setup_logging(config: Dict):
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    log_dir = os.path.join(project_root, config['output']['log_dir'])
    log_file = os.path.join(log_dir, 'training.log')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        f.write('')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int,
                    loss: float, config: Dict, tokenizer_path: str):
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    checkpoint_dir = os.path.join(project_root, config['output']['checkpoint_dir'])
    model_name = config['output']['model_name']

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'tokenizer_path': tokenizer_path,
    }

    torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_name}_latest.pt'))
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch}.pt'))
    logging.info(f"Saved checkpoint at epoch {epoch}")

# ------------- Generation -------------
def sample_next_token(logits: torch.Tensor, temperature: float) -> int:
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1).item()
    return int(next_token)

def generate_sample(model: nn.Module, tokenizer: Tokenizer,
                    device: torch.device, config: Dict, prompt: str = "") -> str:
    model.eval()
    gen = config['generation']
    max_len = gen['max_length']
    temperature = gen['temperature']

    # Encode prompt (post-processor will add <START>/<END>)
    if prompt:
        token_ids = tokenizer.encode(prompt).ids
    else:
        token_ids = [tokenizer.token_to_id("<START>")]

    with torch.no_grad():
        for _ in range(max_len):
            inp = torch.tensor([token_ids], dtype=torch.long, device=device)
            output = model(inp)
            logits = output[0, -1, :]
            next_token = sample_next_token(logits, temperature)
            token_ids.append(next_token)

            # Stop if we see <END>
            if next_token == tokenizer.token_to_id("<END>"):
                break

    # Remove the leading <START> if present before decoding
    start_id = tokenizer.token_to_id("<START>")
    if len(token_ids) > 0 and token_ids[0] == start_id:
        token_ids = token_ids[1:]
    return tokenizer.decode(token_ids)

# ------------- Train/Eval -------------
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer,
                scheduler, criterion, device: torch.device, config: Dict) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()
        output = model(input_ids)  # (B, T-1, V)
        loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
        loss.backward()

        clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        if batch_idx % config['training']['log_frequency'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')

    return total_loss / max(1, num_batches)

def evaluate(model: nn.Module, dataloader: DataLoader, criterion,
             device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            output = model(input_ids)
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

# ------------- Main -------------
def main():
    # Use your existing external config path
    config_path = "/home/joseph_woodall/workspace/reasoning_models/src/training_pipeline/model_config.yml"
    config = load_config(config_path)

    set_seed(42)
    setup_directories(config)
    setup_logging(config)

    logging.info("Starting transformer training (BPE tokenizer)...")
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Device
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['hardware']['device'])
    logging.info(f"Using device: {device}")

    # Data path
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', config['data']['cache_dir'])
    batch_size_for_loading = config['data'].get('load_batch_size', 1000)

    texts = load_text_data(
        cache_dir,
        config['data']['max_files'],
        config['data']['min_text_length'],
        batch_size=batch_size_for_loading
    )
    if not texts:
        raise ValueError("No text data found!")

    # Split data
    split_idx = int(len(texts) * config['data']['train_split'])
    split_idx = max(1, split_idx)  # ensure at least one training text
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:] if split_idx < len(texts) else [texts[-1]]

    logging.info(f"Train texts: {len(train_texts)}, Validation texts: {len(val_texts)}")

    # Tokenizer: train or load
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    tokenizer_path = os.path.join(project_root, config['output'].get('tokenizer_path', 'output/checkpoints/tokenizer.json'))
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

    if not os.path.exists(tokenizer_path):
        logging.info(f"Training BPE tokenizer (vocab={config['model']['src_vocab_size']})...")
        tokenizer = train_bpe_tokenizer(
            train_texts,
            vocab_size=config['model']['src_vocab_size'],
            save_path=tokenizer_path
        )
        logging.info(f"Tokenizer trained and saved to {tokenizer_path}")
    else:
        logging.info(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)

    vocab_size = tokenizer.get_vocab_size()
    # Mirror to tgt vocab (shared LM head)
    config['model']['src_vocab_size'] = vocab_size
    config['model']['tgt_vocab_size'] = vocab_size

    # Datasets
    seq_len = config['data']['sequence_length']
    train_dataset = TextDataset(train_texts, tokenizer, seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, seq_len)

    # DataLoaders
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

    # Model
    model = create_model(config, vocab_size)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer / Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
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

    # Loss: no padding needed since sequences are fixed-length
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    train_loss = 0.0

    for epoch in range(config['training']['epochs']):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, config)

        # Evaluate
        if epoch % config['evaluation']['eval_frequency'] == 0:
            val_loss = evaluate(model, val_loader, criterion, device)
            perplexity = math.exp(min(20, val_loss))  # clamp to avoid inf during early training
            logging.info(f'Epoch {epoch}/{config["training"]["epochs"]}:')
            logging.info(f'  Train Loss: {train_loss:.4f}')
            logging.info(f'  Val   Loss: {val_loss:.4f}')
            logging.info(f'  PPL       : {perplexity:.2f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, tokenizer_path)

            # Sample
            sample = generate_sample(model, tokenizer, device, config, "The")
            logging.info(f'Sample: {sample[:200]}...')

        # Periodic save
        if epoch % config['training']['save_frequency'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, config, tokenizer_path)

        logging.info(f'Epoch {epoch} completed in {time.time() - epoch_start:.2f}s')

    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f}s")

    # Final save
    save_checkpoint(model, optimizer, scheduler, config['training']['epochs'], train_loss, config, tokenizer_path)
    logging.info("Training finished!")

if __name__ == "__main__":
    main()
