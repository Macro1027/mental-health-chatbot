import os
import torch
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline, T5Tokenizer, T5ForConditionalGeneration
from utilities.utils import get_emotions
from peft import PeftConfig, PeftModel

TEXTGEN_MODEL = "Macro27/mental_health"
SENTIMENT_MODEL = "cross-encoder/nli-roberta-base"

class MentalHealthChatbot:
    def __init__(self, debugging=False):
        self.debugging = debugging
        if self.debugging:
            print("Debugging is enabled.")
        self.emotions = get_emotions()
        self.textgen_model, self.textgen_tokenizer = self.load_textgen_model()
        self.sentiment_model, self.sentiment_tokenizer = self.load_sentiment_model()

    def load_textgen_model(self):
        if self.debugging:
            print("Loading text generation model from Huggingface Hub.")
        base_model = "mistralai/Mistral-7B-Instruct-v0.2"
        adapter = "GRMenon/mental-health-mistral-7b-instructv0.2-finetuned-V2"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            add_bos_token=True,
            trust_remote_code=True,
            padding_side='left'
        )

        # Create peft model using base_model and finetuned adapter
        config = PeftConfig.from_pretrained(adapter)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                    device_map='auto',
                                                    torch_dtype='auto')
        model = PeftModel.from_pretrained(model, adapter)

        return model, tokenizer

    def load_sentiment_model(self):
        if self.debugging:
            print("Loading sentiment analysis model.")
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        return model, tokenizer

    def analyze_sentiment(self, input_text):
        # Prepare input pairs in batch
        labels = [(input_text, f"This emotion is {label}.") for label in self.emotions]
        
        # Tokenize inputs in batch
        inputs = self.sentiment_tokenizer(labels, return_tensors='pt', truncation=True, padding=True)

        # Run the model
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)

        # Convert logits to probabilities
        probs = torch.softmax(outputs.logits, dim=1)
        sentiments_idx = torch.topk(probs[:, 2], 3).indices
        sentiments = [self.emotions[idx] for idx in sentiments_idx]

        if self.debugging:
            print(f"Sentiment analysis result: {sentiments}")

        return sentiments

    def cbt_response(self, input_text):
        sentiments = self.analyze_sentiment(input_text)

        prompt = f"{input_text}. I am feeling {', '.join(sentiments)}."

        if self.debugging:
            print(f"Prompt: {prompt}")

        inputs = self.textgen_tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.textgen_model.generate(inputs, max_length=256)
        
        response = self.textgen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response

    def ask(self, user_input):
        response = self.cbt_response(user_input)
        return response
        
if __name__ == "__main__":
    bot = MentalHealthChatbot(debugging=True)
    prompt = "What is depression?"
    print(bot.ask(prompt))
    # model = T5ForConditionalGeneration.from_pretrained(TEXTGEN_MODEL)
    # tokenizer = T5Tokenizer.from_pretrained(TEXTGEN_MODEL)