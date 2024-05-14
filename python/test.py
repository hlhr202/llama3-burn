import torch
import transformers
from transformers import AutoTokenizer
model = "stas/tiny-random-llama-2"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float32,
    device=torch.device("mps")
)

sequences = pipeline(
    'Hello, ',
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=20,
)

print("\n\n\n")

for seq in sequences:
    print(f"Result: {seq['generated_text']}")

