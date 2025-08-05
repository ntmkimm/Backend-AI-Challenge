from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
torch.cuda.empty_cache()

device = "cuda:6"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
model.eval()


def paraphrase(question):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt",
        padding="longest",
        max_length=128,
        truncation=True,
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=128,
            repetition_penalty=2.0,
            num_beams=5,
            num_beam_groups=5,
            num_return_sequences=5,
            no_repeat_ngram_size=4,
            diversity_penalty=1.5,
            trust_remote_code=True
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

print("Inference model...")
import time
start_time = time.time()
text = 'a girl is buying a bouquet'

print(paraphrase(text))
end_time = time.time()
print("Time for inference: ", end_time - start_time)

while True:
    start_time = time.time()
    text = input("Please type the text for paraphrase: ")
    print(paraphrase(text))
    end_time = time.time()
    print("Time for inference: ", end_time - start_time)    