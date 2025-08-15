# server.py
"""
Minimal MCP server for Grok text generation (FastMCP Cloud/Claude Desktop)
"""

import os
import time
from fastmcp import FastMCP
from openai import OpenAI
from typing import Dict, Any, List

mcp = FastMCP(name="Grok")
client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.getenv("XAI_API_KEY"),
)

@mcp.tool
def generate_text(
    prompt: str,
    model: str = "grok-4",
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    grounding: bool = True,  # enables Grok Live Search when True
    delay_seconds: float = 45,
) -> str:
    """
    Generate text with Grok.
    """
    try:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": max_output_tokens,
        }
        if grounding:
            kwargs["extra_body"] = {"search_parameters": {"mode": "auto"}}

        resp = client.chat.completions.create(**kwargs)

        if resp and resp.choices:
            msg = resp.choices[0].message
            if msg and getattr(msg, "content", None):
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                return msg.content

        out: List[str] = []
        for ch in (resp.choices or []):
            msg = getattr(ch, "message", None)
            if msg and getattr(msg, "content", None):
                out.append(msg.content)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        return "".join(out)

    except Exception as e:
        return f"Error generating text: {e}"

if __name__ == "__main__":
    mcp.run()
