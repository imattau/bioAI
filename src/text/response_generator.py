"""Grounded response synthesis from accepted long-term memories."""

from __future__ import annotations

import re


class OllamaResponseGenerator:
    def __init__(self, model: str = "qwen2.5-coder:1.5b"):
        self.model = model
        self.last_mode = "unused"

    @staticmethod
    def _content_words(text: str) -> set[str]:
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "of", "to",
            "in", "on", "at", "for", "and", "or", "that", "this", "it",
            "from", "with", "through", "by", "as", "which", "what", "how",
            "does", "do", "did", "can", "could", "toward", "towards",
            "according", "stored", "memory", "answer",
        }
        return {
            word for word in re.findall(r"[a-z0-9]+", text.lower())
            if word not in stop_words and not word.isdigit()
        }

    @classmethod
    def is_grounded(cls, response: str, question: str,
                    memories: list[dict]) -> bool:
        allowed = cls._content_words(question)
        for memory in memories:
            allowed |= cls._content_words(memory["text"])
        response_words = cls._content_words(response)
        unsupported = response_words - allowed
        return len(unsupported) <= 2

    @staticmethod
    def extractive_fallback(memories: list[dict]) -> str:
        return " ".join(
            f"According to [memory:{memory['id']}], {memory['text']}"
            for memory in memories
        )

    def __call__(self, question: str, memories: list[dict]) -> str:
        import ollama

        sources = "\n".join(
            f"[memory:{memory['id']}] {memory['text']}"
            for memory in memories
        )
        response = ollama.chat(self.model, messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied memories. Do not add facts "
                    "that are absent from them. Cite every factual claim using "
                    "the corresponding [memory:N] marker. If the memories do "
                    "not answer the question, say you do not know."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nMemories:\n{sources}",
            },
        ], options={"temperature": 0.0, "num_predict": 150})
        content = (
            response["message"]["content"]
            if isinstance(response, dict) else response.message.content
        )
        content = content.strip()
        if not self.is_grounded(content, question, memories):
            self.last_mode = "extractive_fallback"
            return self.extractive_fallback(memories)
        for memory in memories:
            marker = f"[memory:{memory['id']}]"
            if marker not in content:
                content = f"{content} {marker}".strip()
        self.last_mode = "grounded_generation"
        return content
