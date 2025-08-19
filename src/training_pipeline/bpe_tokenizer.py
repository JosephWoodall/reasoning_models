from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import BertProcessing
from pathlib import Path

def train_bpe_tokenizer(texts, vocab_size=32000, special_tokens=None, save_path="tokenizer.json"):
    """
    Train a Byte-Pair Encoding (BPE) tokenizer on your dataset.
    """
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]

    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # Train on your texts
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add post-processor to ensure special tokens wrap sequences
    tokenizer.post_processor = BertProcessing(
        ("<END>", tokenizer.token_to_id("<END>")),
        ("<START>", tokenizer.token_to_id("<START>")),
    )

    # Save tokenizer
    tokenizer.save(save_path)
    return tokenizer
