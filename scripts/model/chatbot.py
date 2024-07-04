import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

TEXTGEN_MODEL = "distilgpt2"
SENTIMENT_MODEL = "cross-encoder/nli-roberta-base"
NER_MODEL = "dslim/bert-large-NER"
NER_THRESHOLD = 0.6

def get_emotions():
    with open("data/external/emotions.txt", "r") as data:
        emotions = data.read().replace("/", "\n").strip().replace("\n\n", "\n")
        emotions = emotions.split()
    return emotions

class MentalHealthChatbot:
    def __init__(self, debugging=False):
        self.debugging = debugging
        if self.debugging:
            print("Debugging is enabled.")
        self.emotions = get_emotions()
        self.textgen_model, self.textgen_tokenizer, self.sentiment_model, self.sentiment_tokenizer, self.ner_model, self.ner_tokenizer = self.initialize_models()

    def initialize_models(self):
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)

        # Load text gen model for response generation
        textgen_model_path = os.path.join(model_dir, "textgen_model")
        textgen_tokenizer_path = os.path.join(model_dir, "textgen_tokenizer")

        if os.path.exists(textgen_model_path) and os.path.exists(textgen_tokenizer_path):
            textgen_model = AutoModelForCausalLM.from_pretrained(textgen_model_path, low_cpu_mem_usage=True)
            textgen_tokenizer = AutoTokenizer.from_pretrained(textgen_tokenizer_path, low_cpu_mem_usage=True)
            if self.debugging:
                print(f"Loaded textgen model and tokenizer from local storage {TEXTGEN_MODEL}.")
        else:
            textgen_model = AutoModelForCausalLM.from_pretrained(TEXTGEN_MODEL)
            textgen_tokenizer = AutoTokenizer.from_pretrained(TEXTGEN_MODEL, low_cpu_mem_usage=True)
            textgen_model.save_pretrained(textgen_model_path)
            textgen_tokenizer.save_pretrained(textgen_tokenizer_path)
            if self.debugging:
                print(f"Downloaded and saved textgen model and tokenizer {TEXTGEN_MODEL}.")

        # Load model for sentiment analysis
        sentiment_model_path = os.path.join(model_dir, "sentiment_model")
        sentiment_tokenizer_path = os.path.join(model_dir, "sentiment_tokenizer")

        if os.path.exists(sentiment_model_path) and os.path.exists(sentiment_tokenizer_path):
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
            sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_tokenizer_path)
            if self.debugging:
                print(f"Loaded sentiment model and tokenizer from local storage {SENTIMENT_MODEL}.")
        else:
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
            sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
            sentiment_model.save_pretrained(sentiment_model_path)
            sentiment_tokenizer.save_pretrained(sentiment_tokenizer_path)
            if self.debugging:
                print(f"Downloaded and saved sentiment model and tokenizer {SENTIMENT_MODEL}.")

        # Load NER model
        ner_model_path = os.path.join(model_dir, "ner_model")
        ner_tokenizer_path = os.path.join(model_dir, "ner_tokenizer")

        if os.path.exists(ner_model_path) and os.path.exists(ner_tokenizer_path):
            ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
            ner_tokenizer = AutoTokenizer.from_pretrained(ner_tokenizer_path)
            if self.debugging:
                print(f"Loaded NER model and tokenizer from local storage {NER_MODEL}.")
        else:
            ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
            ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
            ner_model.save_pretrained(ner_model_path)
            ner_tokenizer.save_pretrained(ner_tokenizer_path)
            if self.debugging:
                print(f"Downloaded and saved NER model and tokenizer {NER_MODEL}.")

        return textgen_model, textgen_tokenizer, sentiment_model, sentiment_tokenizer, ner_model, ner_tokenizer

    def generate_response(self, input_text):
        if self.debugging:
            print(f"Generating response for input: {input_text}")

        # Tokenize input text
        inputs = self.textgen_tokenizer(input_text, return_tensors='pt')

        # Generate outputs
        outputs = self.textgen_model.generate(**inputs, max_length=150)

        # Decode the token ids back into text
        response = self.textgen_tokenizer.decode(outputs[0], skip_special_tokens=True)

        if self.debugging:
            print(f"Generated response: {response}")

        return response

    def analyze_sentiment(self, input_text):
        if self.debugging:
            print(f"Analyzing sentiment for text: {input_text}")

        # Prepare the input pairs
        labels = []
        for label in self.emotions:
            hypothesis = f"This emotion is {label}."
            labels.append((input_text, hypothesis))

        # Tokenize the inputs
        inputs = self.sentiment_tokenizer(labels, return_tensors='pt', truncation=True, padding=True)

        # Run the model
        outputs = self.sentiment_model(**inputs)

        # Convert logits to probabilities
        probs = torch.softmax(outputs.logits, dim=1)
        entailment_probs = probs[:, 2]

        # Find the label with the highest probability
        sentiments_idx = torch.topk(entailment_probs, 3).indices
        sentiments = [self.emotions[idx] for idx in sentiments_idx]

        if self.debugging:
            print(f"Sentiment analysis result: {sentiments}")

        return sentiments

    def recognize_entities(self, input_text):
        if self.debugging:
            print(f"Recognizing entities in text: {input_text}")

        entity_recognition = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer, device=0)
        entities = entity_recognition(input_text)
        entities = self.format_NER(entities)

        if self.debugging:
            print(f"Recognized entities: {entities}")

        return entities

    def format_NER(self, ner_results):
        # Organize results by entity type
        entities_by_type = {}
        for entity in ner_results:
            if entity['score'] > NER_THRESHOLD:
                entity_type = entity['entity'].split('-')[-1]  # Get the entity type (e.g., 'PER' from 'B-PER')
                entity_text = entity['word']
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity_text)

        # Map entity types to meaningful names
        entity_type_map = {
            "PER": "People",
            "LOC": "Locations",
            "ORG": "Organizations",
            "MISC": "Miscellaneous"
        }

        # Format the results into a string
        formatted_results = []
        for entity_type, entities in entities_by_type.items():
            if entities:  # Exclude categories with no entries
                meaningful_name = entity_type_map.get(entity_type, entity_type)
                formatted_results.append(f"{meaningful_name}: {', '.join(entities)}")

        result_string = "; ".join(formatted_results)
        return result_string

    def cbt_response(self, input_text):
        if self.debugging:
            print(f"Generating CBT response for input: {input_text}")

        # Analyze sentiment
        sentiments = self.analyze_sentiment(input_text)

        # Recognize entities
        entities = self.recognize_entities(input_text)

        # Define the prompt with CBT context
        prompt = f"""
        You are a mental health chatbot that uses Cognitive Behavioral Therapy (CBT) techniques to help users. Respond to the following input with appropriate CBT techniques:
        Based on this context:
        The user feels {sentiments}, recognized entities are {entities},
        ---------------

        Answer the query: {input_text},
        """

        # Generate response
        response = self.generate_response(prompt)

        top_emotions = ' and '.join(sentiments[:2])
        print(f"I recognize that you are feeling {top_emotions}")
        print(response)

        return response

    def provide_cbt_techniques(self, input_text):
        if self.debugging:
            print(f"Providing CBT techniques for input: {input_text}")

        if "worthless" in input_text or "can't do anything right" in input_text:
            return random.choice([
                "Let's examine that thought. What evidence do you have for and against it?",
                "Can you think of a time when you did something well?"
            ])
        return None

    def guide_mindfulness_exercises(self, input_text):
        if self.debugging:
            print(f"Guiding mindfulness exercises for input: {input_text}")

        if "stress" in input_text or "anxiety" in input_text:
            return random.choice([
                "Let's try a deep breathing exercise. Inhale slowly for 4 seconds, hold for 4 seconds, and exhale for 4 seconds.",
                "Close your eyes and focus on your breath. Notice the sensation of the air entering and leaving your nostrils."
            ])
        return None

    def manage_crisis_situations(self, input_text):
        if self.debugging:
            print(f"Managing crisis situation for input: {input_text}")

        if "suicide" in input_text or "self-harm" in input_text:
            return "It sounds like you're going through a really tough time. Please reach out to a crisis hotline or a trusted person immediately."
        return None

    def handle_rlhf(self, feedback):
        if self.debugging:
            print(f"Handling RLHF with feedback: {feedback}")
        pass

    def ask(self, user_input):
        if self.debugging:
            print(f"Received user input: {user_input}")

        response = self.cbt_response(user_input)
        print(response)

        # Provide CBT techniques if applicable
        cbt_response = self.provide_cbt_techniques(user_input)
        if cbt_response:
            response = cbt_response
            print(response)

        # Guide mindfulness exercises if applicable
        mindfulness_response = self.guide_mindfulness_exercises(user_input)
        if mindfulness_response:
            response = mindfulness_response
            print(response)

        # Manage crisis situations if detected
        crisis_response = self.manage_crisis_situations(user_input)
        if crisis_response:
            response = crisis_response
            print(response)

        # Collect user feedback
        feedback = input("Was this response helpful? (yes/no): ")

        # Handle RLHF
        self.handle_rlhf(feedback)

if __name__ == "__main__":
    bot = MentalHealthChatbot()
    prompt = "My name is Wolfgang and I live in Berlin. I work at Google."
    bot.cbt_response(prompt)
