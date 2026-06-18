from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os

app = FastAPI()

client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

MODEL = "Qwen/Qwen2.5-3B-Instruct"


class Request(BaseModel):
    jobTitle: str
    company: str
    jobDescription: str


def build_prompt(jobTitle, company, jobDescription):
    return f"""
You are a resume expert.

Return ONLY valid JSON.

Job Title: {jobTitle}
Company: {company}

Job Description:
{jobDescription}

Rules:
- Use only given information
- Do not add fake experience
- Make ATS friendly resume
- Keep output strict JSON format
"""


def generate(prompt):
    response = client.text_generation(
        model=MODEL,
        prompt=prompt,
        max_new_tokens=900,
        temperature=0.3
    )
    return response


@app.post("/generate")
def generate_resume(req: Request):

    prompt = build_prompt(req.jobTitle, req.company, req.jobDescription)

    result = generate(prompt)

    return {
        "resume": result
    }
