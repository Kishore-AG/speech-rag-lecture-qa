import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import CHUNKS_FILE, SUMMARIZATION_CONFIG

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_CONFIG["model"], trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    SUMMARIZATION_CONFIG["model"],
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation="eager"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_text(instruction, text_content, max_tokens):
    """Uses the model's native chat template for strict instruction following."""
    # Format exactly how Phi-3 expects to be spoken to
    messages = [
        {"role": "user", "content": f"{instruction}\n\nText:\n{text_content}"}
    ]
    
    # Let the tokenizer wrap the prompt in the correct <|user|> tags
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=3800).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=False,              # Keep it strictly factual
            repetition_penalty=1.0,       # FIX: Set to 1.0 (Off) so it stops hallucinating weird synonyms
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id
        )
        
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

if __name__ == "__main__":
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        full_text = " ".join([c["text"] for c in json.load(f)])

    tokens = tokenizer(full_text)["input_ids"]
    chunk_size = 2500
    sections = [tokenizer.decode(tokens[i:i + chunk_size], skip_special_tokens=True) 
                for i in range(0, len(tokens), chunk_size)]

    print(f"\nTranscript split into {len(sections)} massive section(s). Summarizing...")

    # The strict rules you provided
    rules = """Write a clear and factual summary of the discussion.
Rules:
- Use only information present in the transcript
- Do not add new ideas
- Focus on the main topics discussed
- Ignore filler conversation
- Keep the summary concise"""

    section_summaries = []
    for i, section in enumerate(sections):
        print(f"  -> Processing section {i+1}/{len(sections)}...")
        instruction = f"You are summarizing a portion of a lecture transcript.\n\n{rules}"
        section_summaries.append(generate_text(instruction, section, max_tokens=150))

    print("\nGenerating summary...")
    combined_summaries = "\n\n".join(section_summaries)
    
    final_instruction = f"You are creating a final comprehensive summary from lecture transcript notes.\n\n{rules}"
    final_result = generate_text(final_instruction, combined_summaries, max_tokens=250)

    print("\n" + "="*60)
    print("FINAL LECTURE SUMMARY:")
    print("="*60)
    print(final_result)
    print("="*60)