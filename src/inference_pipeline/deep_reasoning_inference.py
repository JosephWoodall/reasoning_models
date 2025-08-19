import sys
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model import Transformer  # your model class


# -----------------------------
# Tokenizer (kept compatible)
# -----------------------------
class TextTokenizer:
    """Simple word-level tokenizer for text data (compatible with training)."""

    def __init__(self, special_tokens: Dict[str, str]):
        # Accept either training-style keys or inference-style keys
        # Normalize to training keys internally.
        canonical = {
            "pad_token": special_tokens.get("pad_token") or special_tokens.get("pad") or "<PAD>",
            "unk_token": special_tokens.get("unk_token") or special_tokens.get("unk") or "<UNK>",
            "start_token": special_tokens.get("start_token") or special_tokens.get("bos_token") or "<START>",
            "end_token": special_tokens.get("end_token") or special_tokens.get("eos_token") or "<END>",
        }
        self.special_tokens = canonical
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000):
        # Not used at inference if we load from checkpoint; kept for completeness.
        for tok in self.special_tokens.values():
            if tok not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[tok] = idx
                self.id_to_word[idx] = tok

        from collections import Counter
        import re
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            word_counts.update(words)

        for word, _ in word_counts.most_common(max_vocab_size - len(self.special_tokens)):
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        unk_id = self.word_to_id[self.special_tokens['unk_token']]
        return [self.word_to_id.get(w, unk_id) for w in words]

    def decode(self, token_ids: List[int]) -> str:
        words = [self.id_to_word.get(int(t), self.special_tokens['unk_token']) for t in token_ids]
        # Light detokenization for nicer output
        text = ' '.join(words)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'\s+\'\s+', '\'', text)
        return text


# -----------------------------
# Model loading (robust)
# -----------------------------
def load_model(checkpoint_path: Path, device: str = 'cpu') -> Tuple[nn.Module, TextTokenizer, Dict]:
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=torch.serialization.pickle)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint loaded successfully")

    # Restore tokenizer from checkpoint vocab (as in training script) 
    training_specials = {
        'pad_token': '<PAD>',
        'unk_token': '<UNK>',
        'start_token': '<START>',
        'end_token': '<END>'
    }
    tokenizer = TextTokenizer(training_specials)

    if 'word_to_id' in checkpoint and 'id_to_word' in checkpoint:
        tokenizer.word_to_id = checkpoint['word_to_id']
        tokenizer.id_to_word = {int(k): v for k, v in checkpoint['id_to_word'].items()}
        tokenizer.vocab_size = len(tokenizer.word_to_id)
        print(f"Vocab restored: {tokenizer.vocab_size} tokens")
    else:
        raise ValueError("Checkpoint lacks vocabulary (word_to_id/id_to_word)")

    # Pull model config (works with your training format) 
    cfg = checkpoint.get('config', {})
    model_cfg = cfg.get('model', cfg) if isinstance(cfg, dict) else {}
    model_params = {
        'd_model': model_cfg.get('d_model', 128),
        'num_heads': model_cfg.get('num_heads', 4),
        'num_layers': model_cfg.get('num_layers', 2),
        'd_ff': model_cfg.get('d_ff', 512),
        'dropout': model_cfg.get('dropout', 0.1),
        'max_len': model_cfg.get('max_len', 1024),
    }
    print(f"Creating model with params: {model_params}")

    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=model_params['d_model'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        d_ff=model_params['d_ff'],
        dropout=model_params['dropout'],
        max_len=model_params.get('max_len', 1024)
    )
    print("Loading model state...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print(f"Model ready on {device}")
    return model, tokenizer, model_params


# -----------------------------
# Sampling utilities
# -----------------------------
def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Apply top-k and/or nucleus (top-p) filtering."""
    logits = logits.clone()

    if top_k > 0:
        topk_vals, _ = torch.topk(logits, top_k)
        min_topk = topk_vals[..., -1].unsqueeze(-1)
        logits[logits < min_topk] = -float('inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff = (cumulative > top_p).float()
        # shift so we always keep at least one token
        cutoff[..., 0] = 0.0
        sorted_logits[cutoff.bool()] = -float('inf')
        # restore original order
        logits = torch.full_like(logits, -float('inf'))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return logits


def try_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Support both of your observed signatures:
      - training used: output = model(x) → logits  ()
      - inference (older version) used: model(x, x, mask) ()
    """
    try:
        return model(input_ids)  # training-time API
    except TypeError:
        # Fall back to encoder/decoder style
        if hasattr(model, "create_look_ahead_mask"):
            mask = model.create_look_ahead_mask(input_ids.size(1)).to(input_ids.device)
        else:
            mask = None
        return model(input_ids, input_ids, mask)


def find_token_id(tokenizer: TextTokenizer, token_str: str) -> Optional[int]:
    return tokenizer.word_to_id.get(token_str, None)


# -----------------------------
# Tiny safe calculator tool
# -----------------------------
import ast
import operator as op

_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg, ast.FloorDiv: op.floordiv
}

def _safe_eval_expr(node):
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_expr(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_expr(node.left), _safe_eval_expr(node.right))
    raise ValueError("Unsafe expression")

def safe_calc(expr: str) -> str:
    try:
        node = ast.parse(expr, mode='eval').body
        return str(_safe_eval_expr(node))
    except Exception as e:
        return f"CALC_ERROR({e})"


# -----------------------------
# Core generation (single pass)
# -----------------------------
@torch.no_grad()
def generate_once(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    device: str = 'cpu',
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    stop_on_end_token: bool = True,
    tool_use: bool = True,
) -> str:
    """
    One sampled continuation with optional tool-use turn taking: CALC[ ... ]
    """
    # bootstrap with START if available
    start_id = find_token_id(tokenizer, tokenizer.special_tokens['start_token'])
    end_id = find_token_id(tokenizer, tokenizer.special_tokens['end_token'])

    # Structured prompt to encourage reasoning
    base_prompt = prompt

    # Tool-use loop: if model writes CALC[expr], we evaluate and append the result.
    # We interleave generation chunks to allow tool calls mid-stream.
    text_context = base_prompt
    token_ids = tokenizer.encode(text_context)
    if start_id is not None and (len(token_ids) == 0 or token_ids[0] != start_id):
        token_ids = [start_id] + token_ids

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        logits = try_forward(model, input_ids)[0, -1, :] / max(temperature, 1e-5)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1,)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if stop_on_end_token and end_id is not None and int(next_token) == end_id:
            break

        # After each full decode step, check if the latest text contains a CALC[...] tool call.
        if tool_use:
            decoded = tokenizer.decode(input_ids[0].tolist())
            # robustly find the last unmatched CALC[ ... ]
            m = list(re.finditer(r"CALC\[(.*?)\]", decoded, flags=re.DOTALL))
            if m:
                expr = m[-1].group(1)
                # If expression appears to be newly completed, append result to the context
                result = safe_calc(expr)
                # We append a compact result token sequence
                append_text = f"\nResult: {result}\n"
                append_ids = tokenizer.encode(append_text)
                input_ids = torch.cat([input_ids, torch.tensor([append_ids], device=device)], dim=1)

        if input_ids.size(1) >= max_tokens:
            break

    return tokenizer.decode(input_ids[0].tolist())


# -----------------------------
# Chain-of-Thought wrapper
# -----------------------------
def cot_prompt(user_prompt: str) -> str:
    return (
        f"Question: {user_prompt}\n"
        f"Let's think step by step. Show your reasoning as a list of steps.\n"
        f"Then on a new line, write: Answer: <final answer here>"
    )


# -----------------------------
# Self-consistency (vote)
# -----------------------------
def extract_final_answer(text: str) -> str:
    # Look for "Answer: ..." line; fallback to last non-empty line
    for line in text.splitlines()[::-1]:
        line = line.strip()
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
        if line:
            fallback = line
    return fallback if 'fallback' in locals() else text.strip()


def self_consistency(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    device: str = 'cpu',
    samples: int = 8,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Tuple[str, List[str], Counter]:
    """Run N independent CoT samples and majority-vote the final answers."""
    cot = cot_prompt(prompt)
    generations = []
    answers = []

    for _ in range(samples):
        g = generate_once(
            model, tokenizer, cot, device=device,
            max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
            tool_use=True
        )
        generations.append(g)
        answers.append(extract_final_answer(g))

    tally = Counter(answers)
    best_answer, _ = tally.most_common(1)[0]
    return best_answer, generations, tally


# -----------------------------
# Optional: simple beam search
# -----------------------------
@torch.no_grad()
def beam_search(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    device: str = 'cpu',
    beam_size: int = 4,
    max_tokens: int = 200
) -> str:
    """
    Log-likelihood beam search (no CoT), kept simple for clarity.
    """
    start_id = find_token_id(tokenizer, tokenizer.special_tokens['start_token'])
    end_id = find_token_id(tokenizer, tokenizer.special_tokens['end_token'])

    init_ids = tokenizer.encode(prompt)
    if start_id is not None and (len(init_ids) == 0 or init_ids[0] != start_id):
        init_ids = [start_id] + init_ids

    beams = [(torch.tensor([init_ids], device=device, dtype=torch.long), 0.0)]  # (tokens, logprob)

    for _step in range(max_tokens):
        new_beams = []
        for ids, logp in beams:
            if end_id is not None and ids[0, -1].item() == end_id:
                new_beams.append((ids, logp))
                continue

            logits = try_forward(model, ids)[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_lp, topk_idx = torch.topk(log_probs, k=beam_size)

            for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                new_seq = torch.cat([ids, torch.tensor([[idx]], device=device)], dim=1)
                new_beams.append((new_seq, logp + lp))

        # prune
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        # early stop if all ended
        if end_id is not None and all(b[0][0, -1].item() == end_id for b in beams):
            break

    best_ids, _ = max(beams, key=lambda x: x[1])
    return tokenizer.decode(best_ids[0].tolist())


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    # Default to your existing checkpoint path; edit as needed. 
    checkpoint_path = Path("/home/joseph_woodall/workspace/reasoning_models/output/checkpoints/transformer_lm_latest.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print("="*64)
        print("Loading model...")
        model, tokenizer, _ = load_model(checkpoint_path, device=device)
        print("="*64)

        while True:
            print("\nEnter a prompt (or /quit):")
            prompt = input().strip()
            if prompt.lower() in {"/q", "/quit", "exit"}:
                break

            print("\nMode? [1] CoT (single)  [2] Self-Consistency  [3] Beam  [4] Plain sample")
            mode = input().strip() or "2"

            if mode == "1":
                text = generate_once(
                    model, tokenizer, cot_prompt(prompt), device=device,
                    max_tokens=256, temperature=0.9, top_k=50, top_p=0.9, tool_use=True
                )
                print("\n--- Generation (CoT) ---\n")
                print(text)

            elif mode == "2":
                best, gens, tally = self_consistency(
                    model, tokenizer, prompt, device=device,
                    samples=8, max_tokens=256, temperature=1.0, top_k=50, top_p=0.9
                )
                print("\n--- Voted Answer ---\n")
                print(best)
                print("\n--- Vote counts ---")
                for ans, c in tally.most_common():
                    print(f"{c}  |  {ans}")
                print("\n--- Raw generations (for inspection) ---")
                for i, g in enumerate(gens, 1):
                    print(f"\n[{i}] --------------------\n{g}")

            elif mode == "3":
                text = beam_search(model, tokenizer, prompt, device=device, beam_size=4, max_tokens=200)
                print("\n--- Generation (Beam) ---\n")
                print(text)

            else:
                text = generate_once(
                    model, tokenizer, prompt, device=device,
                    max_tokens=200, temperature=0.8, top_k=50, top_p=0.9, tool_use=True
                )
                print("\n--- Generation (Plain) ---\n")
                print(text)

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
