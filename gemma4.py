"""
Sample: Gemma 4 E4B (4-billion parameter) inference with Hugging Face Transformers
Requirements:
    pip install transformers torch accelerate bitsandbytes
    HF_TOKEN env var must be set (model is gated — accept license at huggingface.co/google/gemma-4-4b-it)
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-26B-A4B-it"
HF_TOKEN = os.getenv("HF_TOKEN")  # export HF_TOKEN=hf_...


def load_model(quantize: bool = True):
    """Load Gemma 4 E4B, optionally in 4-bit quantization to fit on consumer GPUs."""
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if quantize
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",          # spreads across available GPU/CPU
        torch_dtype=torch.bfloat16 if not quantize else None,
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def generate(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Run a single inference pass and return the assistant reply."""
    messages = [{"role": "user", "content": prompt}]

    # apply_chat_template formats the prompt with <start_of_turn> tags
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    print(f"Loading {MODEL_ID} ...")
    tokenizer, model = load_model(quantize=True)
    print("Model ready.\n")

    examples = [
        "Explain quantum entanglement in simple terms.",
        "Write a Python function that reverses a linked list.",
        "What are three tips for writing clean code?",
    ]

    for prompt in examples:
        print(f"Prompt: {prompt}")
        reply = generate(tokenizer, model, prompt)
        print(f"Reply : {reply}\n{'-' * 60}\n")


if __name__ == "__main__":
    main()
