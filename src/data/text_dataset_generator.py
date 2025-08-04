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


# Example usage
if __name__ == "__main__":
    # Create dataset generator
    generator = TextDatasetGenerator(cache_dir="text_cache")

    # URL for the Gutenberg text you specified
    gutenberg_url = "https://www.gutenberg.org/cache/epub/25983/pg25983.txt"

    # Create dataset
    print("Creating dataset from Project Gutenberg text...")
    dataset = generator.create_dataset(
        url=gutenberg_url,
        output_file="gutenberg_dataset.json",
        chunk_size=1000,
        overlap=100
    )

    # Print statistics
    stats = generator.get_dataset_stats(dataset)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nDataset saved to: gutenberg_dataset.json")
    print(f"First chunk preview:")
    print(f"{'='*50}")
    print(dataset['chunks'][0][:300] + "..." if len(dataset['chunks']
          [0]) > 300 else dataset['chunks'][0])
