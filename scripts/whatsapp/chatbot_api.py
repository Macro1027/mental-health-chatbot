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
    bot = ChatbotAPI("meta-llama/Meta-Llama-3-70B-Instruct")
    print(bot.query({'inputs': "write me a python program to reverse a list", "task": "text-generation"})[0]['generated_text'])