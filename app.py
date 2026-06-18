import streamlit as st
from huggingface_hub import InferenceClient

generator = InferenceClient(
    provider="hf-inference",
    api_key=st.secrets["HF_TOKEN"]
)

critic = InferenceClient(
    provider="hf-inference",
    api_key=st.secrets["HF_TOKEN"]
)

refiner = InferenceClient(
    provider="hf-inference",
    api_key=st.secrets["HF_TOKEN"]
)


GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CRITIC_MODEL = "microsoft/Phi-4-mini-instruct"
REFINER_MODEL = "google/gemma-3-4b-it"


def call(model, prompt):
    res = generator.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return res.choices[0].message.content


st.title("Multi Model AI")

user_input = st.text_area("Enter prompt")

if st.button("Run"):

    draft = call(GEN_MODEL, user_input)

    review = call(CRITIC_MODEL, f"Review this:\n{draft}")

    final = call(REFINER_MODEL, f"""
Improve this answer:

{draft}

Feedback:
{review}
""")

    st.write(final)
