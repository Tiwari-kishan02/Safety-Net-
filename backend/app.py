from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open('models/spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return "SafetyNet+ backend is running!"

@app.route('/recovery')
def recovery():
    return render_template('recovery.html')

@app.route('/message-check', methods=['GET', 'POST'])
def message_check():
    result = None
    if request.method == 'POST':
        message = request.form['message']
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        result = "This looks like a SCAM message! Be careful." if prediction == 1 else "This message looks SAFE."
    return render_template('message_check.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
app = Flask(__name__)

@app.route('/')
def home():
    return "SafetyNet+ backend is running!"

@app.route('/recovery')
def recovery():
    return render_template('recovery.html')

if __name__ == '__main__':
    app.run(debug=True)
