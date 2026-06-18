import streamlit as st
from huggingface_hub import InferenceClient

client = InferenceClient(api_key=st.secrets["HF_TOKEN"])

GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CRITIC_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REFINER_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def call(model, prompt):
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return res.choices[0].message.content


st.title("Multi Model AI")

user_input = st.text_area("Enter prompt")

if st.button("Run"):

    draft = call(GEN_MODEL, user_input)

    review = call(CRITIC_MODEL, "Review and fix this:\n" + draft)

    final = call(REFINER_MODEL, f"""
Improve this answer:

{draft}

Review:
{review}
""")

    st.write(final)
