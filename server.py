from fastapi import FastAPI
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

app = FastAPI()

# 🔥 CONFIG (speed tuning)
MODEL_NAME = "facebook/m2m100_418M"
DEVICE = "cpu"   # GPU ho to "cuda"
MAX_LENGTH = 100  # 👈 input limit (speed boost)

print("Loading model...")

tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

# 🔥 CPU optimization
torch.set_num_threads(4)   # apne CPU ke hisaab se adjust karo

print("Model loaded!")

# 🔥 Warmup (first request slow hota hai, isse fix)
with torch.no_grad():
    tokenizer.src_lang = "en"
    dummy = tokenizer("hello", return_tensors="pt").to(DEVICE)
    model.generate(**dummy)

class RequestData(BaseModel):
    text: str
    target: str
    source: str = "en"  # 👈 default

@app.post("/translate")
def translate(data: RequestData):
    text = data.text[:MAX_LENGTH]  # 🔥 limit for speed
    target = data.target
    source = data.source

    tokenizer.src_lang = source

    # 🔥 encode
    encoded = tokenizer(text, return_tensors="pt").to(DEVICE)

    # 🔥 NO GRAD (big speed boost)
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(target),
            max_length=128,     # limit output
            num_beams=1         # 👈 faster (default 5 hota hai)
        )

    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return {
        "translated": result[0],
        "detected_source": source
    }