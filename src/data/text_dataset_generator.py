"""
Text Dataset Generator

A class for generating text datasets by scraping content from web sources,
specifically designed for Project Gutenberg texts.
"""

import requests
import re
import os
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import json
import yaml
import time


class TextDatasetGenerator:
    """
    A class to generate text datasets by scraping and processing text content
    from web sources, with specific support for Project Gutenberg texts.
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the TextDatasetGenerator.

        Args:
            cache_dir (str): Directory to cache downloaded texts
        """
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; TextDatasetGenerator/1.0)'
        })

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_text(self, url: str, force_refresh: bool = False) -> str:
        """
        Fetch text content from a URL with caching support.

        Args:
            url (str): The URL to fetch text from
            force_refresh (bool): Whether to bypass cache and re-download

        Returns:
            str: The raw text content

        Raises:
            requests.RequestException: If the request fails
        """
        # Generate cache filename from URL
        parsed_url = urlparse(url)
        cache_filename = f"{parsed_url.netloc}_{parsed_url.path.replace('/', '_')}.txt"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # Check cache first (unless force refresh is requested)
        if not force_refresh and os.path.exists(cache_path):
            print(f"Loading from cache: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()

        # Download the text
        print(f"Downloading text from: {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        # Decode content with proper encoding
        text_content = response.content.decode('utf-8', errors='ignore')

        # Cache the downloaded text
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(text_content)

        return text_content

    def clean_gutenberg_text(self, raw_text: str) -> str:
        """
        Clean Project Gutenberg text by removing headers, footers, and metadata.

        Args:
            raw_text (str): Raw text from Project Gutenberg

        Returns:
            str: Cleaned text content
        """
        # Common Gutenberg start markers
        start_markers = [
            r'\*\*\* START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK',
            r'\*\*\*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*'
        ]

        # Common Gutenberg end markers
        end_markers = [
            r'\*\*\* END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*',
            r'END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK',
            r'\*\*\*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*'
        ]

        text = raw_text

        # Find and remove everything before the start marker
        for pattern in start_markers:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[match.end():]
                break

        # Find and remove everything after the end marker
        for pattern in end_markers:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                text = text[:match.start()]
                break

        # Additional cleaning
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove page numbers and chapter markers that might remain
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        return text.strip()

    def split_into_chunks(self, text: str, chunk_size: int = 1000,
                          overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks for dataset creation.

        Args:
            text (str): The text to split
            chunk_size (int): Target size of each chunk in characters
            overlap (int): Number of characters to overlap between chunks

        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If we're not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(end - 200, start)
                sentence_end = -1

                for i in range(end, search_start, -1):
                    if text[i] in '.!?' and i + 1 < len(text) and text[i + 1].isspace():
                        sentence_end = i + 1
                        break

                if sentence_end > start:
                    end = sentence_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def create_dataset(self, url: str, output_file: Optional[str] = None,
                       chunk_size: int = 1000, overlap: int = 100,
                       format_type: str = 'json') -> Dict[str, Any]:
        """
        Create a complete text dataset from a URL.

        Args:
            url (str): URL to scrape text from
            output_file (str, optional): Path to save the dataset
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
            format_type (str): Output format ('json' or 'txt')

        Returns:
            Dict[str, Any]: Dataset metadata and statistics
        """
        # Fetch and clean the text
        raw_text = self.fetch_text(url)

        # Clean the text (assuming it's from Gutenberg)
        if 'gutenberg.org' in url.lower():
            cleaned_text = self.clean_gutenberg_text(raw_text)
        else:
            # Basic cleaning for non-Gutenberg sources
            cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', raw_text)
            cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
            cleaned_text = cleaned_text.strip()

        # Split into chunks
        chunks = self.split_into_chunks(cleaned_text, chunk_size, overlap)

        # Create dataset
        dataset = {
            'metadata': {
                'source_url': url,
                'total_characters': len(cleaned_text),
                'total_words': len(cleaned_text.split()),
                'num_chunks': len(chunks),
                'chunk_size': chunk_size,
                'overlap': overlap,
                'created_at': self._get_timestamp()
            },
            'chunks': chunks
        }

        # Save to file if specified
        if output_file:
            if format_type.lower() == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
            elif format_type.lower() == 'txt':
                with open(output_file, 'w', encoding='utf-8') as f:
                    for i, chunk in enumerate(chunks):
                        f.write(f"=== CHUNK {i+1} ===\n")
                        f.write(chunk)
                        f.write(f"\n{'='*50}\n\n")

        return dataset

    def get_dataset_stats(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about a dataset.

        Args:
            dataset (Dict[str, Any]): The dataset to analyze

        Returns:
            Dict[str, Any]: Dataset statistics
        """
        chunks = dataset.get('chunks', [])

        if not chunks:
            return {'error': 'No chunks found in dataset'}

        chunk_lengths = [len(chunk) for chunk in chunks]
        word_counts = [len(chunk.split()) for chunk in chunks]

        stats = {
            'num_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'avg_words_per_chunk': sum(word_counts) / len(word_counts),
            'total_words': sum(word_counts),
            'total_characters': sum(chunk_lengths)
        }

        return stats

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO format string."""
        from datetime import datetime
        return datetime.now().isoformat()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config("/home/joseph_woodall/workspace/reasoning_models/src/data/text_dataset_generator.yml")

    # Pull relevant config settings
    url_pattern = config["url_pattern"]
    start_id = config["book_ids"]["start"]
    end_id = config["book_ids"]["end"]
    cache_dir = config["cache"]["directory"]
    file_ext = config["cache"].get("file_extension", ".txt")
    force_refresh = config["download"].get("force_refresh", False)
    max_retries = config["download"].get("max_retries", 3)
    timeout = config["download"].get("timeout", 30)
    delay = config["download"].get("delay_between_requests", 1)

    # Processing options (not all are used in this snippet, but you can expand)
    min_length = config["processing"].get("min_length", 1000)

    # Initialize generator
    generator = TextDatasetGenerator(cache_dir=cache_dir)

    all_chunks = []
    metadata = []
    errors = []

    for book_id in range(start_id, end_id + 1):
        url = url_pattern.format(id=book_id)
        print(f"Processing book ID {book_id} from {url}")

        retries = 0
        while retries < max_retries:
            try:
                raw_text = generator.fetch_text(url, force_refresh=force_refresh)
                if config["processing"].get("remove_gutenberg_header", True) or config["processing"].get("remove_gutenberg_footer", True):
                    text = generator.clean_gutenberg_text(raw_text)
                else:
                    text = raw_text
                # Apply additional processing if needed...

                if len(text) < min_length:
                    print(f"Book ID {book_id} skipped: text too short ({len(text)})")
                    break

                chunks = generator.split_into_chunks(
                    text,
                    chunk_size=min_length,
                    overlap=100  # adjust as needed
                )
                all_chunks.extend(chunks)
                metadata.append({
                    "book_id": book_id,
                    "source_url": url,
                    "num_chunks": len(chunks),
                    "total_characters": len(text)
                })
                print(f"Book ID {book_id}: {len(chunks)} chunks")
                break  # Success, break out of retries loop

            except Exception as e:
                print(f"Book ID {book_id} failed: {e}")
                retries += 1
                time.sleep(delay)
                if retries == max_retries:
                    errors.append({"book_id": book_id, "url": url, "error": str(e)})

    # Aggregate dataset
    dataset = {
        "metadata": metadata,
        "chunks": all_chunks,
        "errors": errors,
        "config_used": config,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # Save dataset
    with open("gutenberg_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nAggregated dataset saved to gutenberg_dataset.json")
    print(f"Total chunks: {len(all_chunks)} | Books processed: {len(metadata)} | Errors: {len(errors)}")

if __name__ == "__main__":
    main()
