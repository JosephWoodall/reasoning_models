import os
import re
import json
import random
import time
from urllib.parse import urlparse
from datetime import datetime

import requests
import nltk
from nltk.corpus import wordnet

nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

def synonym_augment(text: str, p_replace: float = 0.1) -> str:
    """Randomly replace some nouns/verbs with synonyms."""
    words = nltk.word_tokenize(text)
    new_words = []
    for w in words:
        if random.random() < p_replace:
            syns = wordnet.synsets(w)
            lemmas = [l.name() for s in syns for l in s.lemmas()]
            lemmas = [l for l in lemmas if l.lower() != w.lower()]
            if lemmas:
                w = random.choice(lemmas)
        new_words.append(w)
    return ' '.join(new_words)

def reasoning_prompt(chunk: str) -> str:
    """Wrap chunk with reasoning scaffold."""
    return f"Consider the following passage:\n{chunk}\nExplain step by step your reasoning and then answer the question about this passage.\nAnswer: "

def extract_topics(text: str) -> list:
    """Naive proper noun extraction."""
    words = nltk.word_tokenize(text)
    return list(set([w for w in words if w.istitle() and len(w) > 2]))

class CrossRefReasoningGenerator:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; DatasetGen/1.0)'})
        self.chunk_memory = []

    def fetch_text(self, url: str, force_refresh: bool = False) -> str:
        parsed = urlparse(url)
        cache_file = os.path.join(self.cache_dir, f"{parsed.netloc}_{parsed.path.replace('/', '_')}.txt")
        if os.path.exists(cache_file) and not force_refresh:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        text = resp.content.decode('utf-8', errors='ignore')
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return text

    def clean_text(self, text: str) -> str:
        start_markers = [r'\*\*\* START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*']
        end_markers = [r'\*\*\* END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*']
        for m in start_markers:
            match = re.search(m, text, re.IGNORECASE|re.DOTALL)
            if match: text = text[match.end():]
        for m in end_markers:
            match = re.search(m, text, re.IGNORECASE|re.DOTALL)
            if match: text = text[:match.start()]
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()

    def split_chunks(self, text: str, chunk_size=1000, overlap=100):
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                search_start = max(end-200, start)
                sentence_end = -1
                for i in range(end, search_start, -1):
                    if text[i] in '.!?' and i+1 < len(text) and text[i+1].isspace():
                        sentence_end = i+1
                        break
                if sentence_end > start: end = sentence_end
            chunk = text[start:end].strip()
            if chunk: chunks.append(chunk)
            start = end - overlap
            if start >= len(text): break
        return chunks

    def add_cross_reference(self, chunk: str) -> str:
        if not self.chunk_memory or random.random() > 0.5: return chunk
        prev_chunk = random.choice(self.chunk_memory)
        topics = extract_topics(prev_chunk)
        if not topics: return chunk
        topic = random.choice(topics)
        return chunk + f"\nReference: Earlier we discussed '{topic}'. Explain how it relates to this passage.\nAnswer: "

    def create_dataset(self, book_ids: list, output_file="cross_ref_dataset.json",
                       chunk_size=1000, overlap=100):
        all_chunks, metadata, errors = [], [], []

        for book_id in book_ids:
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            try:
                raw_text = self.fetch_text(url)
                text = self.clean_text(raw_text)
                if len(text) < 500: continue

                chunks = self.split_chunks(text, chunk_size, overlap)
                processed_chunks = []
                for chunk in chunks:
                    chunk = synonym_augment(reasoning_prompt(chunk))
                    chunk = self.add_cross_reference(chunk)
                    processed_chunks.append(chunk)
                    self.chunk_memory.append(chunk)

                all_chunks.extend(processed_chunks)
                metadata.append({'book_id': book_id, 'url': url, 'num_chunks': len(processed_chunks), 'chars': len(text)})
                print(f"Book {book_id}: {len(processed_chunks)} chunks processed")

            except Exception as e:
                errors.append({'book_id': book_id, 'url': url, 'error': str(e)})
                print(f"Error book {book_id}: {e}")
                time.sleep(1)

        dataset = {
            "metadata": metadata,
            "chunks": all_chunks,
            "errors": errors,
            "created_at": datetime.now().replace(microsecond=0).isoformat() + "Z"
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"Dataset saved to {output_file} | Total chunks: {len(all_chunks)}")
        return dataset

if __name__ == "__main__":
    start_book_id = 100
    end_book_id = 800
    print("="*50)
    print(f"Collecting books from {start_book_id} to {end_book_id}")
    print("="*50)
    generator = CrossRefReasoningGenerator(cache_dir="text_cache")
    dataset = generator.create_dataset(book_ids=list(range(start_book_id, end_book_id)))
