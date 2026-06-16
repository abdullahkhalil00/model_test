import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("AI Resume Builder MVP")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

user_input = st.text_area("Enter your prompt", "Write a resume summary for frontend developer")

if st.button("Generate"):
    messages = [
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    st.write(result)
