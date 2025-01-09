"""
title: Qdrant RAG Pipeline
author: your_name
date: 2024-01-10
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from Qdrant vector database
requirements: qdrant-client, langchain, langchain-community, python-dotenv
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from qdrant_client import QdrantClient
import requests

load_dotenv()

class Pipeline:
    class Valves(BaseModel):
        """Configuration for Qdrant Pipeline"""
        QDRANT_API_URL: str = os.getenv("QDRANT_API_URL", "<key trực tiếp>")
        QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "<key trực tiếp>")
        QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "<key trực tiếp>")
        HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "<key trực tiếp>")
        EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(self):
        self.name = "Qdrant RAG Pipeline"
        self.embeddings = None
        self.qdrant_client = None
        self.valves = self.Valves()

    async def on_startup(self):
        """Initialize connections on startup"""
        try:
            print(f"Starting {self.name}...")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=self.valves.HUGGINGFACE_API_KEY,
                model_name=self.valves.EMBEDDINGS_MODEL_NAME
            )
            print("Embeddings model initialized")

            # Initialize Qdrant
            self.qdrant_client = QdrantClient(
                url=self.valves.QDRANT_API_URL,
                api_key=self.valves.QDRANT_API_KEY,
                timeout=10
            )
            print("Qdrant client initialized")

            # Verify collection
            collection_info = self.qdrant_client.get_collection(
                collection_name=self.valves.QDRANT_COLLECTION
            )
            print(f"Connected to collection: {self.valves.QDRANT_COLLECTION}")
            print(f"Vector size: {collection_info.config.params.vectors.size}")

        except Exception as e:
            print(f"Startup error: {str(e)}")
            raise

    async def on_shutdown(self):
        """Cleanup on shutdown"""
        if self.qdrant_client:
            self.qdrant_client.close()

    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> dict:
        """Search Qdrant collection"""
        try:
            url = f"{self.valves.QDRANT_API_URL}/collections/{self.valves.QDRANT_COLLECTION}/points/search"
            headers = {
                "Authorization": f"Bearer {self.valves.QDRANT_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "vector": query_vector,
                "limit": top_k,
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            return {"result": response.json().get("result", [])}
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return {"error": str(e)}

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Process user message and return relevant context"""
        try:
            print(f"Processing: {user_message}")
            
            # Generate embedding
            query_vector = self.embeddings.embed_query(user_message)
            
            # Search Qdrant
            results = self.search_vectors(query_vector)
            if "error" in results:
                return f"Search error: {results['error']}"

            # Format results
            matches = results.get("result", [])
            if not matches:
                return "No relevant information found"

            # Format context
            context = []
            for idx, match in enumerate(matches, 1):
                score = match.get("score", 0)
                content = match.get("payload", {}).get("content", "No content")
                context.append(f"{idx}. [Score: {score:.2f}] {content}")

            return "\n\n".join(context)

        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            return f"Error: {str(e)}"
