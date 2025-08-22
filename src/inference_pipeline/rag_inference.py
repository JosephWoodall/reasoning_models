"""
RAG (Retrieval-Augmented Generation) Inference Pipeline

This implementation:
1. Uses FAISS for efficient similarity search
2. Supports both dense and sparse retrievers
3. Implements basic chunk management and context windowing
4. Provides flexible document storage and retrieval
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
import logging
import numpy as np
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.model import Transformer
from src.inference_pipeline.deep_reasoning_inference import (
    load_model, 
    top_k_top_p_filtering
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document in the knowledge base."""
    id: str
    text: str
    metadata: Optional[Dict] = None
    embedding: Optional[np.ndarray] = None

class DocumentStore:
    """Manages document storage and retrieval."""
    
    def __init__(self, dimension: int = 384):
        """Initialize document store with FAISS index."""
        self.dimension = dimension
        
        base_index = faiss.IndexFlatIP(dimension)
        
        self.index = faiss.IndexIDMap(base_index)
        
        if faiss.get_num_gpus() > 0:
            print(f"FAISS using GPU with dimension {dimension}")
            gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
        else:
            print(f"FAISS using CPU with dimension {dimension}")
            
        self.documents: List[Document] = []
        self.doc_ids_map: Dict[str, int] = {}
    
    def add_document(self, doc: Document) -> None:
        """Add a document to the store."""
        if doc.id in self.doc_ids_map:
            logger.warning(f"Document {doc.id} already exists, updating...")
            idx = self.doc_ids_map[doc.id]
            self.documents[idx] = doc
            if doc.embedding is not None:
                emb = np.ascontiguousarray(doc.embedding.reshape(1, -1))
                self.index.add_with_ids(emb, np.array([idx], dtype=np.int64))
        else:
            idx = len(self.documents)
            self.doc_ids_map[doc.id] = idx
            self.documents.append(doc)
            if doc.embedding is not None:
                emb = np.ascontiguousarray(doc.embedding.reshape(1, -1))
                self.index.add_with_ids(emb, np.array([idx], dtype=np.int64))
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents using FAISS."""
        query_embedding = np.ascontiguousarray(
            F.normalize(torch.from_numpy(query_embedding), p=2, dim=0)
            .numpy()
            .reshape(1, -1)
            .astype('float32')
        )
        
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx != -1:  
                doc_idx = int(idx)  
                if 0 <= doc_idx < len(self.documents):
                    results.append((self.documents[doc_idx], float(score)))
        
        return results

class RAGInference:
    """RAG inference pipeline combining retrieval and generation."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: object,
        doc_store: DocumentStore,
        device: str = 'cuda',
        max_chunk_size: int = 512,
        stride: int = 128,
        max_context_chunks: int = 3
    ):
        """Initialize the RAG pipeline."""
        self.device = device
        self.doc_store = doc_store
        self.max_chunk_size = max_chunk_size
        self.stride = stride
        self.max_context_chunks = max_context_chunks
        
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into a fixed-size embedding using the model's encoder.
        Output: [1, d_model] (L2-normalized)
        """
        self.model.eval()
        with torch.no_grad():
            encoding = self.tokenizer.encode(text)
            token_ids = torch.tensor([encoding.ids], device=self.device)

            src_embedded = self.model.src_embedding(token_ids) * math.sqrt(self.model.d_model)
            src_embedded = self.model.pos_encoding(src_embedded)
            src_embedded = self.model.dropout(src_embedded)
            
            encoder_output = src_embedded
            for layer in self.model.encoder_layers:
                encoder_output = layer(encoder_output)

            mask = token_ids.ne(self.tokenizer.token_to_id("<PAD>")).float().unsqueeze(-1)
            summed = (encoder_output * mask).sum(dim=1)
            denom = mask.squeeze(-1).sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_pooled = summed / denom

            mean_pooled = F.normalize(mean_pooled, p=2, dim=-1)
            return mean_pooled

    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        tokens = self.tokenizer.encode(text).ids
        chunks = []
        
        for i in range(0, len(tokens), self.stride):
            chunk = tokens[i:i + self.max_chunk_size]
            if len(chunk) > self.stride:  
                chunks.append(self.tokenizer.decode(chunk))
        
        return chunks
    
    def add_to_knowledge_base(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32
    ) -> None:
        """Process and add texts to the knowledge base with batched operations."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        if metadata is None:
            metadata = [{}] * len(texts)

        # First, chunk all texts and collect metadata
        all_chunks = []
        all_chunk_ids = []
        all_chunk_metadata = []
        
        print("\nChunking documents...")
        for text_idx, (text, doc_id, meta) in enumerate(zip(texts, ids, metadata)):
            chunks = self._chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_chunk_ids.append(f"{doc_id}_chunk_{chunk_idx}")
                all_chunk_metadata.append({**meta, "original_id": doc_id, "chunk_idx": chunk_idx})
            
            if text_idx % 1000 == 0 and text_idx > 0:
                print(f"Chunked {text_idx}/{len(texts)} documents...")
        
        print(f"\nGenerating embeddings for {len(all_chunks)} chunks in batches...")
        documents_to_add = []
        
        # Process in batches
        for batch_start in range(0, len(all_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(all_chunks))
            batch_chunks = all_chunks[batch_start:batch_end]
            
            # Batch encode all chunks
            batch_embeddings = []
            for chunk in batch_chunks:
                embedding = self._encode_text(chunk).cpu().numpy()
                batch_embeddings.append(embedding)
            
            # Create document objects for the batch
            for chunk, chunk_id, meta, embedding in zip(
                batch_chunks,
                all_chunk_ids[batch_start:batch_end],
                all_chunk_metadata[batch_start:batch_end],
                batch_embeddings
            ):
                doc = Document(
                    id=chunk_id,
                    text=chunk,
                    metadata=meta,
                    embedding=embedding
                )
                documents_to_add.append(doc)
            
            # Add the batch to FAISS
            if len(documents_to_add) >= 1000:  # Bulk add every 1000 documents
                print(f"Adding batch of {len(documents_to_add)} documents to index...")
                self._bulk_add_to_store(documents_to_add)
                documents_to_add = []
            
            if batch_start % (batch_size * 10) == 0:
                print(f"Processed {batch_start}/{len(all_chunks)} chunks...")
        
        # Add any remaining documents
        if documents_to_add:
            print(f"Adding final batch of {len(documents_to_add)} documents to index...")
            self._bulk_add_to_store(documents_to_add)
    
    def _bulk_add_to_store(self, documents: List[Document]) -> None:
        """Add multiple documents to the store at once."""
        # Prepare arrays for FAISS
        embeddings = []
        indices = []
        
        for doc in documents:
            idx = len(self.doc_store.documents)
            self.doc_store.doc_ids_map[doc.id] = idx
            self.doc_store.documents.append(doc)
            
            embeddings.append(doc.embedding)
            indices.append(idx)
        
        # Convert to numpy arrays
        embeddings_array = np.ascontiguousarray(np.vstack(embeddings).astype('float32'))
        indices_array = np.array(indices, dtype=np.int64)
        
        # Add to FAISS in one batch
        self.doc_store.index.add_with_ids(embeddings_array, indices_array)
    
    def _prepare_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """Prepare context from retrieved documents."""
        retrieved_docs = sorted(retrieved_docs, key=lambda x: x[1], reverse=True)
        retrieved_docs = retrieved_docs[:self.max_context_chunks]
        
        context_parts = []
        for doc, score in retrieved_docs:
            context_parts.append(f"{doc.text}")
        
        return "\n\n".join(context_parts)
    
    def generate(
        self,
        query: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Tuple[str, List[Document]]:
        """Generate response using RAG."""
        query_embedding = self._encode_text(query).cpu().numpy()
        
        retrieved_docs = self.doc_store.search(query_embedding)
        
        context = self._prepare_context(retrieved_docs)
        
        full_prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"
        
        input_ids = self.tokenizer.encode(full_prompt).ids
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        generated = []
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits,
                    top_k=top_k,
                    top_p=top_p
                )
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1),
                    num_samples=1
                )
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.token_to_id("<END>"):
                    break
        
        response = self.tokenizer.decode(generated)
        return response, [doc for doc, _ in retrieved_docs]

def main():
    """Interactive RAG inference with fixed paths."""
    
    '''
    Change the model_path to anything found in output/checkpoints. The training portion stores all of the models
    in checkpoints of i =1; for demonstration (and poc) uses, I just loaded the most recent transformer
    I trained, which is called transformer_lm_latest.pt
    
    So you know, do what you want

    You can also bypass this entire process if you want to use something from Hugging Face's model hub. 
    Personally, I like making all of my things from scratch for learning, which is this repo's purpose
    
    But you do you boo boo.
    '''
    model_path = "/home/joseph_woodall/workspace/reasoning_models/output/checkpoints/transformer_lm_latest.pt"
    knowledge_base_path = "/home/joseph_woodall/workspace/reasoning_models/src/data/cross_ref_dataset.json"
    
    print("\nInitializing RAG system...")
    
    model, tokenizer, config = load_model(Path(model_path), 'cuda')
    d_model = model.d_model if hasattr(model, 'd_model') else 384  # fallback to default if not found
    
    doc_store = DocumentStore(dimension=d_model)
    
    rag = RAGInference(
        model=model,
        tokenizer=tokenizer,
        doc_store=doc_store
    )
    
    print("\nLoading knowledge base...")
    with open(knowledge_base_path) as f:
        kb_data = json.load(f)
    
    print("\nData structure check:")
    if isinstance(kb_data, list):
        print(f"Loaded {len(kb_data)} items in a list")
        if len(kb_data) > 0:
            print(f"First item type: {type(kb_data[0])}")
            print("First item preview:", str(kb_data[0])[:200])
    elif isinstance(kb_data, dict):
        print("Loaded a dictionary with keys:", list(kb_data.keys()))
    else:
        print(f"Loaded data of type: {type(kb_data)}")
    
    texts = []
    ids = []
    metadata = []
    
    if isinstance(kb_data, dict) and "chunks" in kb_data:
        chunks = kb_data["chunks"]
        if isinstance(chunks, list):
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    text = chunk.get("text", "")
                    chunk_id = chunk.get("id", f"chunk_{i}")
                    chunk_metadata = chunk.get("metadata", {})
                    
                    if text.strip():
                        texts.append(text)
                        ids.append(chunk_id)
                        metadata.append(chunk_metadata)
                elif isinstance(chunk, str) and chunk.strip():
                    texts.append(chunk)
                    ids.append(f"chunk_{i}")
                    metadata.append({})
    else:
        raise ValueError("Expected a dictionary with 'chunks' key in the knowledge base")
    
    print("Oh dear reader, processing and indexing these below documents might take a while! So strap in number 2!")
    print("(I REALLY need to find a more efficient way to do this, so dont roast me too harshly)")
    print("Because what do we say?? GIVE ME BOOTSTRAP OR GIVE ME DEATH!!")
    print("But yeah, this is O(n), so might be best to store this in db before run (as in, this would be split out and run separately)")
    print(f"\nProcessing and indexing {len(texts)} documents...")
    rag.add_to_knowledge_base(
        texts=texts,
        ids=ids,
        metadata=metadata
    )
    
    print("\nRAG system ready! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            if not query or query.lower() == 'quit':
                print("\nExiting RAG system. Goodbye!")
                break
            
            print("\nProcessing query...")
            response, retrieved = rag.generate(query)
            
            print("\nResponse:")
            print("-" * 50)
            print(response.strip())
            print("-" * 50)
            
            print("\nRetrieved Context Documents:")
            print("-" * 50)
            for i, doc in enumerate(retrieved, 1):
                print(f"\n{i}. Document {doc.id}:")
                preview = doc.text[:200]
                if len(doc.text) > 200:
                    last_space = preview.rfind(" ")
                    if last_space > 0:
                        preview = preview[:last_space] + "..."
                    else:
                        preview += "..."
                print(preview)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\nExiting RAG system. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()
