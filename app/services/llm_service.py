import json

from groq import Groq

from app.core.config import settings
from app.core.logger import logger

MODEL = "llama-3.3-70b-versatile"


class LLMService:

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        timeout: int = 18,
    ) -> dict | None:
        """Return a parsed JSON object, or None if the call/parse fails."""
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                timeout=timeout,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"LLM JSON call failed: {e}")
            return None

    def generate_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: int = 18,
    ) -> str | None:
        """Return generated text, or None if the call fails."""
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                timeout=timeout,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM text call failed: {e}")
            return None

    def generate(self, prompt: str) -> str | None:
        return self.generate_text(system="", user=prompt)
