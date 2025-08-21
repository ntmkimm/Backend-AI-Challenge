from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from config.settings import DEVICE, PARAPHRASE_MODEL

class ParaphraseService:
    def __init__(self):
        print("Init paraphrase service")
        self.tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to(DEVICE)
        self.model.eval()

    def paraphrase(self, question: str):
        input_ids = self.tokenizer(
            f'paraphrase: {question}',
            return_tensors="pt",
            padding="longest",
            max_length=128,
            truncation=True,
        ).input_ids.to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
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

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)