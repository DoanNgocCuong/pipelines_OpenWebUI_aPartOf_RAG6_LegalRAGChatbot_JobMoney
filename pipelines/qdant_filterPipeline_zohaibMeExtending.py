"""
title: Qdrant Vector Search Pipeline
author: Zohaib
date: 2025-01-10
version: 1.0
license: MIT
description: Pipeline for semantic search using Qdrant Cloud
requirements: requests, python-dotenv, llama-index, langchain, langchain-community, qdrant-client
environment_variables: QDRANT_API_URL, QDRANT_API_KEY, QDRANT_COLLECTION, HUGGINGFACE_API_KEY
"""

import os
import json
import asyncio
from typing import List, Union, Optional, Generator, Iterator
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import requests

load_dotenv()

class Pipeline:
    class Valves(BaseModel):
        model_config = {
            "arbitrary_types_allowed": True
        }
        
        pipelines: List[str] = []  # Connected pipelines
        priority: int = 0  # Pipeline priority
        QDRANT_API_URL: str = os.getenv("QDRANT_API_URL", "<>")
        QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "<>")
        QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "<>")
        HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "<>")
        EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(self):
        self.type = "filter"  # or "manifold" for provider
        self.name = "Qdrant Vector Search"
        self.valves = self.Valves()
        self._initialize_clients()

    def _initialize_clients(self):
        try:
            print("Initializing Pipeline...")
            print(f"Using Qdrant URL: {self.valves.QDRANT_API_URL}")
            print(f"Using Collection: {self.valves.QDRANT_COLLECTION}")
            
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
                timeout=10  # Add timeout
            )
            print("Qdrant client initialized")

            # Check collection
            collection_info = self.qdrant_client.get_collection(
                collection_name=self.valves.QDRANT_COLLECTION
            )
            vector_size = collection_info.config.params.vectors.size
            print(f"Collection vector size: {vector_size}")
            
            if vector_size != 768:
                raise ValueError(f"Collection vector size mismatch: expected 768, got {vector_size}")

            print("Successfully connected to Qdrant")
            print("Pipeline initialized successfully")
            
        except Exception as e:
            print(f"Detailed error in initialization: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    async def on_startup(self):
        print(f"Starting {self.name} pipeline...")
        self._initialize_clients()

    async def on_shutdown(self):
        print(f"Shutting down {self.name} pipeline...")
        if hasattr(self, 'qdrant_client'):
            self.qdrant_client.close()

    async def on_valves_updated(self):
        print("Valves configuration updated")
        self._initialize_clients()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Pre-process incoming messages"""
        try:
            if "messages" in body:
                last_message = body["messages"][-1]["content"]
                vector = self.generate_embedding(last_message)
                body["vector"] = vector
            return body
        except Exception as e:
            print(f"Inlet error: {e}")
            return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Post-process outgoing messages"""
        return body

    def generate_embedding(self, query_text: str) -> List[float]:
        """Generate embedding vector"""
        try:
            print(f"Generating embedding for text: {query_text[:100]}...")
            embedding_vector = self.embeddings.embed_query(query_text)
            print(f"Generated vector dimension: {len(embedding_vector)}")
            print(f"First few values: {embedding_vector[:5]}")
            return embedding_vector
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> dict:
        """Search Qdrant collection with 768-dimensional vectors"""
        try:
            # Debug info
            print(f"Vector dimension: {len(query_vector)}")
            print(f"First few values: {query_vector[:5]}")
            
            # Construct URL and headers directly
            url = f"{self.valves.QDRANT_API_URL}/collections/{self.valves.QDRANT_COLLECTION}/points/search"
            headers = {
                "Authorization": f"Bearer {self.valves.QDRANT_API_KEY}",
                "Content-Type": "application/json",
            }
            
            # Construct payload
            payload = {
                "vector": query_vector,
                "limit": top_k,
            }
            
            print("Sending request to Qdrant...")
            print(f"URL: {url}")
            print(f"Payload size: {len(str(payload))}")
            
            # Use requests instead of qdrant_client
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            print(f"Response Status: {response.status_code}")
            results = response.json()
            print(f"Got {len(results.get('result', []))} results")
            
            return {"result": results.get("result", [])}
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e.response.status_code} {e.response.text}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Error in search_vectors: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

    def _format_results(self, matches: list) -> str:
        """Format search results for display"""
        results = []
        for idx, match in enumerate(matches, 1):
            # Adapt to raw API response format
            score = match.get("score", 0)
            payload = match.get("payload", {})
            content = payload.get("content", "No content")
            results.append(f"{idx}. [Score: {score:.2f}] {content}")
        return "\n\n".join(results)

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, dict]:
        """Main pipeline processing"""
        try:
            print(f"Processing message: {user_message}")
            
            # Generate embedding
            query_vector = self.generate_embedding(user_message)
            if not query_vector:
                return {"response": "Failed to generate embedding"}

            # Search vectors
            results = self.search_vectors(query_vector)
            if "error" in results:
                return {"response": f"Search error: {results['error']}"}

            # Format results
            matches = results.get("result", [])
            if not matches:
                return {"response": "No relevant results found"}

            # Format response for UI display
            formatted_response = {
                "response": self._format_results(matches),
                "matches": [match.dict() for match in matches],
                "total": len(matches)
            }
            
            print(f"Returning response: {formatted_response}")
            return formatted_response
            
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            return {"response": f"Error: {str(e)}"}

    def _check_collection(self):
        """Verify collection exists and check vector configuration"""
        try:
            collection_info = self.qdrant_client.get_collection(
                collection_name=self.valves.QDRANT_COLLECTION
            )
            vector_size = collection_info.config.params.vectors.size
            print(f"Collection vector size: {vector_size}")
            
            if vector_size != 768:
                print(f"Warning: Collection expects {vector_size}-dimensional vectors")
                return False
            
            return True
        except Exception as e:
            print(f"Error checking collection: {e}")
            return False
