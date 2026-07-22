import json
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

MODEL = "gemini-3.5-flash-lite"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    document_id: str
    text: str
    schema: dict


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/extract")
def extract(req: ExtractRequest):

    prompt = f"""
You are an expert invoice extraction engine.

Extract structured information from the invoice below.

Invoice Text:

{req.text}

Return JSON that EXACTLY follows this JSON Schema.

JSON Schema:

{json.dumps(req.schema, indent=2)}

IMPORTANT RULES

- Return ONLY valid JSON.
- No markdown.
- No explanation.
- No extra keys.
- Include every required key.
- Use null if information is missing.
- Dates must be YYYY-MM-DD.
- Numbers must be JSON numbers.
- Booleans must be true/false.
- Arrays must preserve original order.
- Strings should match the document unless normalization is explicitly required by the schema.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    text = response.text.strip()

    # Remove markdown fences
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Extract JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        text = match.group(0)

    try:
        result = json.loads(text)
    except Exception:
        result = {}

    return result
