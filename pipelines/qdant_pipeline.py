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


class Pipeline:
    class Valves(BaseModel):
        QDRANT_API_URL: str
        QDRANT_API_KEY: str
        QDRANT_COLLECTION: str

    def __init__(self):
        self.valves = self.Valves(
            **{
                "QDRANT_API_URL": os.getenv("QDRANT_API_URL", "https://fcbf96b5-0f95-47b1-b088-dd1eba2a2758.us-east4-0.gcp.cloud.qdrant.io:6333"),
                "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", "WbQ_8KeZKchBfQ-atnt5zfbkIShw6slMNvF0PK8qIOEIgaYqTyZLmw"),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "cmc_final_db"),
            }
        )

    async def on_startup(self):
        print("Qdrant Cloud Pipeline started.")

    async def on_shutdown(self):
        print("Qdrant Cloud Pipeline stopped.")

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

        # Convert the user message to a query vector (dummy example, replace with real embedding logic)
        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]  # Replace with embedding generation logic

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
