from __future__ import annotations

import json
from typing import Any


def keyword_to_complex_prompt(query_text: str) -> str:
    return f"""You are provided with the following keywords query: {query_text}
Use the keywords to generate a search query for a search engine.
The search query should be in natural language and should be of high complexity and length.
This is an example of a search query: "Find me a dataset about the impact of climate change on agriculture in Europe"

The answer should be a single string, without any additional text or explanation.
"""


def background_doc_prompt(query_text: str) -> str:
    return f"""
Generate a background document from web to answer the given question. {query_text}

No additional text is needed, just the background document.
"""


def subquery_decomposition_prompt(query_text: str, background_document: str) -> str:
    return f"""
Address the following query based on the contexts provided. Identify any missing information or areas
requiring validation, especially if time-sensitive data is involved. Then, formulate several specific search engine
queries to acquire or validate the necessary knowledge

Query: {query_text}
Context: {background_document}

Output the queries you generate in a numbered list, no other text is required.
"""


def rerank_prompt(query_text: str, candidates: list[dict[str, Any]]) -> str:
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    return f"""
Rank the candidate datasets by relevance to the user query.
You must use only dataset IDs that appear in the candidates list.
Return a JSON object with this exact schema:
{{
  "ranking": [
    {{
      "dataset_id": "string",
      "score": 0.000,
      "reason": "short rationale"
    }}
  ]
}}

Rules:
1. Include every candidate exactly once in "ranking".
2. Sort from most relevant to least relevant.
3. score must be between 0 and 1 with up to 3 decimals.
4. Output only valid JSON and no markdown fences.

User query:
{query_text}

Candidates:
{candidates_json}
"""

