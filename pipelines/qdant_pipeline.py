"""
title: Qdrant Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from Qdrant using the Haystack library.
"""

from haystack import Pipeline as HaystackPipeline
from haystack.components.embedders import SentenceTransformersEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from typing import List, Union, Generator, Iterator
import asyncio

__all__ = ['Pipeline']

class Pipeline:
    def __init__(self):
        self.basic_rag_pipeline = None
        self._initialized = False

    async def on_startup(self):
        # Initialize components
        text_embedder = SentenceTransformersEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        retriever = InMemoryEmbeddingRetriever()
        prompt_builder = PromptBuilder(
            template="Given the context: {documents}, answer the question: {question}"
        )
        llm = OpenAIGenerator()

        # Create pipeline
        self.basic_rag_pipeline = HaystackPipeline()
        self.basic_rag_pipeline.add_component("text_embedder", text_embedder)
        self.basic_rag_pipeline.add_component("retriever", retriever)
        self.basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
        self.basic_rag_pipeline.add_component("llm", llm)

        # Connect components
        self.basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.basic_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.basic_rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

    async def ensure_initialized(self):
        if not self._initialized:
            await self.on_startup()
            self._initialized = True

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Ensure pipeline is initialized
        asyncio.run(self.ensure_initialized())
        
        print(messages)
        print(user_message)

        question = user_message
        response = self.basic_rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
            }
        )

        return response["llm"]["replies"][0]
