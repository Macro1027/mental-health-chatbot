from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime

app = Flask(__name__)

# Sample responses for the chatbot
responses = {
    "anxious": "It's okay to feel anxious sometimes. How can I help you?",
    "happy": "That's great to hear! What made you feel happy?",
    "sad": "I'm sorry to hear that you're feeling sad. I'm here to listen.",
    "stressed": "Stress can be tough. Would you like to talk about it?",
    "lonely": "Feeling lonely is hard. Remember, you're not alone."
}

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'emotions.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class UserEmotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(45), nullable=False)
    emotion = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, nullable=False)

    def __repr__(self):
        return f'<UserEmotion {self.user_id} - {self.emotion} on {self.date}>'

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    # Simple logic to determine response based on user input
    response = responses.get(user_input.lower(), "I'm not sure how to respond to that.")
    return jsonify({'response': response})

@app.route('/mood')
def mood():
    return render_template('moodtracker.html')

@app.route('/submit-emotion', methods=['POST'])
def submit_emotion():
    data = request.json
    user_id = 1
    emotion = data['emotion']
    date = datetime.strptime(data['date'], '%Y-%m-%d')

    new_emotion = UserEmotion(user_id=user_id, emotion=emotion, date=date)
    
    db.session.add(new_emotion)
    db.session.commit()

    return jsonify({"status": "success"}), 201


@app.route('/get-emotions', methods=['GET'])
def get_emotions():
    user_id = request.args.get('user_id')

    # Fetch emotions for the current month
    print(user_id)
    results = UserEmotion.query.filter_by(user_id=user_id).filter(UserEmotion.date >= datetime.now().replace(day=1)).all()
    
    # Format results for response
    emotions_count = {}
    for emotion in results:
        if emotion.emotion in emotions_count:
            emotions_count[emotion.emotion] += 1
        else:
            emotions_count[emotion.emotion] = 1

    return jsonify(emotions_count)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
