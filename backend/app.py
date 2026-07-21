from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load message detection model
with open('models/spam_model.pkl', 'rb') as f:
    message_model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    message_vectorizer = pickle.load(f)

# Load email detection model
with open('models/email_model.pkl', 'rb') as f:
    email_model = pickle.load(f)
with open('models/email_vectorizer.pkl', 'rb') as f:
    email_vectorizer = pickle.load(f)

# Load scam numbers database
scam_db = pd.read_csv('datasets/scam_numbers.csv')
scam_db['number'] = scam_db['number'].astype(str)

# Red flag keywords/phrases commonly used in Indian scams
RED_FLAGS = [
    "pay a fee", "pay the fee", "processing fee", "registration fee",
    "pay rs", "pay inr", "deposit amount", "security deposit",
    "selected for", "selection confirmed", "job confirmed",
    "send documents", "share your documents", "submit documents on",
    "verify your account", "account will be blocked", "account suspended",
    "click here immediately", "click the link below", "confirm within 24 hours",
    "otp to confirm", "share your otp", "provide your otp",
    "congratulations you have been selected", "limited seats",
    "urgent action required", "kyc update", "update your kyc",
    "bank account will be blocked", "railway recruitment fee",
    "pay to confirm your seat", "processing charge"
]

def check_red_flags(text):
    text_lower = text.lower()
    found_flags = [flag for flag in RED_FLAGS if flag in text_lower]
    return found_flags

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recovery')
def recovery():
    return render_template('recovery.html')

@app.route('/message-check', methods=['GET', 'POST'])
def message_check():
    result = None
    if request.method == 'POST':
        message = request.form['message']
        message_vec = message_vectorizer.transform([message])
        ml_prediction = message_model.predict(message_vec)[0]
        flags = check_red_flags(message)

        if ml_prediction == 1 or len(flags) > 0:
            result = "This looks like a SCAM message! Be careful."
            if flags:
                result += " (Red flags found: " + ", ".join(flags) + ")"
        else:
            result = "This message looks SAFE."
    return render_template('message_check.html', result=result)

@app.route('/email-check', methods=['GET', 'POST'])
def email_check():
    result = None
    if request.method == 'POST':
        email_text = request.form['email_text']
        email_vec = email_vectorizer.transform([email_text])
        ml_prediction = email_model.predict(email_vec)[0]
        flags = check_red_flags(email_text)

        if ml_prediction == 1 or len(flags) > 0:
            result = "This looks like a FAKE/SCAM email! Be careful."
            if flags:
                result += " (Red flags found: " + ", ".join(flags) + ")"
        else:
            result = "This email looks GENUINE."
    return render_template('email_check.html', result=result)

@app.route('/number-check', methods=['GET', 'POST'])
def number_check():
    result = None
    if request.method == 'POST':
        number = request.form['number'].strip()
        match = scam_db[scam_db['number'] == number]
        if not match.empty:
            category = match.iloc[0]['category']
            reports = match.iloc[0]['reports']
            result = f"WARNING: This number is reported as SCAM! Category: {category}, Reports: {reports}"
        else:
            result = "This number has no scam reports in our database. Still be cautious."
    return render_template('number_check.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
