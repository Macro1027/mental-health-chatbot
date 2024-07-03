import requests
import jsonify

class Messenger:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.app = Flask(__name__)
    
    def init_facebook(self):
        @self.app.route('/webhook', methods=['POST'])
        def webhook():
            data = requests.request.json
            user_input = data['message']['text']
            response = self.chatbot.ask(user_input)
            return jsonify({'text': response})
    
    def init_whatsapp(self):
        @self.app.route('/whatsapp_webhook', methods=['POST'])
        def whatsapp_webhook():
            data = requests.request.json
            user_input = data['message']['text']
            phone_number = data['message']['from']
            response = self.chatbot.ask(user_input)
            self.send_whatsapp_message(phone_number, response)
            return jsonify({'status': 'success'})

    def send_whatsapp_message(self, phone_number, message):
        key = None
        url = "https://api.whatsapp.com/send"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        data = {
            "phone": phone_number,
            "body": message
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()