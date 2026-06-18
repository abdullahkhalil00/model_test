import streamlit as st
from huggingface_hub import InferenceClient

HF_TOKEN = st.secrets["HF_TOKEN"]

generator = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

critic = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

refiner = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)


GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CRITIC_MODEL = "microsoft/Phi-4-mini-instruct"
REFINER_MODEL = "google/gemma-3-4b-it"


def call_model(model, prompt):
    response = generator.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=700
    )

    return response.choices[0].message.content


def generate_answer(user_question):

    generator_prompt = f"""
Answer the user's question clearly and accurately.

Question:
{user_question}
"""

    draft_answer = call_model(
        GENERATOR_MODEL,
        generator_prompt
    )

    critic_prompt = f"""
You are an expert reviewer.

Question:
{user_question}

Draft Answer:
{draft_answer}

Tasks:
1. Find factual mistakes.
2. Find missing information.
3. Find unclear explanations.
4. Suggest improvements.

Return a detailed review.
"""

    review = call_model(
        CRITIC_MODEL,
        critic_prompt
    )

    refiner_prompt = f"""
You are an expert editor.

User Question:
{user_question}

Original Draft:
{draft_answer}

Reviewer Feedback:
{review}

Create the best final answer.

Requirements:
- Correct mistakes.
- Add missing information.
- Improve clarity.
- Keep the answer concise.
"""

    final_answer = call_model(
        REFINER_MODEL,
        refiner_prompt
    )

    return {
        "draft": draft_answer,
        "review": review,
        "final": final_answer
    }


st.title("Multi-Agent AI Assistant")

question = st.text_area(
    "Ask anything"
)

if st.button("Generate") and question:

    with st.spinner("Thinking..."):

        result = generate_answer(question)

        st.subheader("Final Answer")
        st.write(result["final"])

        with st.expander("Draft Answer"):
            st.write(result["draft"])

        with st.expander("Critic Review"):
            st.write(result["review"])
