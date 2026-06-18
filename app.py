import streamlit as st
from huggingface_hub import InferenceClient, model_info

st.title("Multi Model AI updated")

# --- one-time debug helper: uncomment to see which models your account can actually use ---
# for m in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct-1M", "Qwen/Qwen2.5-1.5B-Instruct"]:
#     info = model_info(m, expand="inferenceProviderMapping")
#     st.write(m, "->", info.inference_provider_mapping)

client = InferenceClient(api_key=st.secrets["HF_TOKEN"])

GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CRITIC_MODEL = "Qwen/Qwen2.5-7B-Instruct"
REFINER_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def call(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Surface the real provider error instead of letting Streamlit redact it
        st.error(f"Call to {model} failed:\n\n{e}")
        st.stop()


def run_pipeline(user_input):
    draft = call(GEN_MODEL, user_input)

    review = call(CRITIC_MODEL, f"""
Review this answer:
{draft}
Find mistakes and improvements.
""")

    final = call(REFINER_MODEL, f"""
Improve this answer using feedback.
Answer:
{draft}
Feedback:
{review}
Return final clean answer.
""")

    return draft, review, final


user_input = st.text_area("Enter prompt")

if st.button("Run") and user_input:
    draft, review, final = run_pipeline(user_input)

    st.subheader("Final Answer")
    st.write(final)

    with st.expander("Draft"):
        st.write(draft)

    with st.expander("Review"):
        st.write(review)
