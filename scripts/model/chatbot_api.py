import requests

class ChatbotAPI:
    def __init__(self, model):
        token = "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"
        self.url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {
            "Authorization": f"Bearer {token}"
        }

    def query(self, payload):
        response = requests.post(self.url, headers=self.headers, json=payload)
        return response.json()
    

if __name__ == "__main__":
    bot = ChatbotAPI("google/gemma-2b")
    inputs = "hello, how do I overcome depression"
    print(bot.query({'inputs': inputs})[0]['generated_text'])