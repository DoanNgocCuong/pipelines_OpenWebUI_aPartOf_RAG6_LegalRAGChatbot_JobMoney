"""
title: Qdrant Cloud Search Pipeline
author: YourName
date: 2025-01-10
version: 1.0
license: MIT
description: A pipeline to interact with Qdrant Cloud for vector search.
requirements: requests
"""

import os
import requests
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class Pipeline:
    class Valves(BaseModel):
        QDRANT_API_URL: str
        QDRANT_API_KEY: str
        QDRANT_COLLECTION: str
        HUGGINGFACE_API_KEY: str
        EMBEDDINGS_MODEL_NAME: str

    def __init__(self):
        self.valves = self.Valves(
            **{
                "QDRANT_API_URL": os.getenv("QDRANT_API_URL", "https://your-qdrant-url"),
                "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", "your-qdrant-api-key"),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "your-collection-name"),
                "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY", "your-huggingface-api-key"),
                "EMBEDDINGS_MODEL_NAME": os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            }
        )

        # Initialize Hugging Face embeddings
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name=self.valves.EMBEDDINGS_MODEL_NAME,
            api_key=self.valves.HUGGINGFACE_API_KEY,
            model_kwargs={'device': 'auto'}
        )

    async def on_startup(self):
        print("Qdrant Cloud Pipeline started.")

    async def on_shutdown(self):
        print("Qdrant Cloud Pipeline stopped.")

    def generate_embedding(self, query_text: str) -> List[float]:
        """
        Generate embedding vector for a given query text using Hugging Face API.
        """
        print("Generating embedding for query...")
        embedding_vector = self.embeddings.embed_query(query_text)
        print(f"Generated embedding: {embedding_vector[:5]}...")  # Log first 5 values
        return embedding_vector

    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> dict:
        """
        Search Qdrant collection for nearest neighbors to the query vector.
        """
        url = f"{self.valves.QDRANT_API_URL}/collections/{self.valves.QDRANT_COLLECTION}/points/search"
        headers = {
            "Authorization": f"Bearer {self.valves.QDRANT_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "vector": query_vector,
            "limit": top_k,
        }

        try:
            print("Testing Qdrant search...")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()  # Assuming API returns JSON
        except requests.exceptions.RequestException as e:
            print(f"Error querying Qdrant: {e}")
            return {"error": "Unable to query Qdrant Cloud"}

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Process user message and query Qdrant for vector search.
        """
        print(f"User message: {user_message}")

        # Generate embedding vector
        query_vector = self.generate_embedding(user_message)

        # Search in Qdrant
        qdrant_response = self.search_vectors(query_vector)

        # Process response
        if "error" in qdrant_response:
            return qdrant_response["error"]

        # Format the results
        results = qdrant_response.get("result", [])
        if not results:
            return "No relevant data found in Qdrant Cloud."

        formatted_results = "\n".join([f"- ID: {item['id']}, Score: {item['score']}" for item in results])
        return f"Here are the top results from Qdrant Cloud:\n\n{formatted_results}"

# Example usage
if __name__ == "__main__":
    pipeline = Pipeline()
    user_query = "Find information about deep learning and AI models."
    results = pipeline.pipe(user_query, model_id="test-model", messages=[], body={})
    print(results)
