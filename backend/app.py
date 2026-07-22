from flask import Flask, render_template, request
import pickle
import pandas as pd
from PIL import Image
import pytesseract

app = Flask(__name__)

with open('models/spam_model.pkl', 'rb') as f:
    message_model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    message_vectorizer = pickle.load(f)

with open('models/email_model.pkl', 'rb') as f:
    email_model = pickle.load(f)
with open('models/email_vectorizer.pkl', 'rb') as f:
    email_vectorizer = pickle.load(f)

scam_db = pd.read_csv('datasets/scam_numbers.csv')
scam_db['number'] = scam_db['number'].astype(str)

fraud_accounts_db = pd.read_csv('datasets/fraud_accounts.csv')
fraud_accounts_db['account_number'] = fraud_accounts_db['account_number'].astype(str)

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

SUSPICIOUS_DOMAIN_PATTERNS = [
    "sbi-verify", "rbi-alert", "kbc-lottery", "bank-update",
    "verify-account", "secure-login", "-support", "kyc-update"
]

def check_red_flags(text):
    text_lower = text.lower()
    return [flag for flag in RED_FLAGS if flag in text_lower]

def analyze_phone_number(raw_number):
    number = raw_number.strip().replace(" ", "").replace("-", "")
    if number.startswith("+91"):
        core = number[3:]
        if core.isdigit() and len(core) == 10:
            return check_indian_number(core)
        return "INVALID: Please enter a valid 10-digit number after +91."
    elif number.startswith("91") and len(number) == 12 and number.isdigit():
        core = number[2:]
        return check_indian_number(core)
    elif number.startswith("+"):
        code = number[1:3]
        return f"FOREIGN NUMBER: This number appears to belong to a foreign country code (+{code}), not India. Exercise extra caution with international numbers, as they are commonly used in international scam calls."
    elif number.isdigit() and len(number) == 10:
        return check_indian_number(number)
    else:
        return "INVALID INPUT: Please enter a valid 10-digit Indian number (with or without +91)."

def check_indian_number(core):
    match = scam_db[scam_db['number'] == core]
    if not match.empty:
        category = match.iloc[0]['category']
        reports = match.iloc[0]['reports']
        return f"WARNING: This is an Indian (+91) number reported as SCAM! Category: {category}, Reports: {reports}"
    else:
        return "This is a valid Indian (+91) number format with no scam reports in our database. Still be cautious."

@app.route('/')
def home():
    return render_template('home.html')

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

@app.route('/email-check-image', methods=['POST'])
def email_check_image():
    image_result = None
    file = request.files.get('email_image')
    if file:
        img = Image.open(file)
        extracted_text = pytesseract.image_to_string(img)
        if extracted_text.strip():
            email_vec = email_vectorizer.transform([extracted_text])
            ml_prediction = email_model.predict(email_vec)[0]
            flags = check_red_flags(extracted_text)
            if ml_prediction == 1 or len(flags) > 0:
                image_result = "This looks like a FAKE/SCAM email! Be careful."
                if flags:
                    image_result += " (Red flags found: " + ", ".join(flags) + ")"
            else:
                image_result = "This email looks GENUINE."
        else:
            image_result = "Could not read text from the image. Try a clearer screenshot."
    return render_template('email_check.html', image_result=image_result)

@app.route('/email-address-check', methods=['POST'])
def email_address_check():
    address_result = None
    sender_email = request.form['sender_email'].strip().lower()
    if '@' not in sender_email or '.' not in sender_email.split('@')[-1]:
        address_result = "SUSPICIOUS: This does not look like a valid email address format."
    else:
        domain = sender_email.split('@')[1]
        found = [pattern for pattern in SUSPICIOUS_DOMAIN_PATTERNS if pattern in domain]
        if found:
            address_result = f"SUSPICIOUS: Domain matches known fake-bank/scam patterns: {', '.join(found)}"
        else:
            address_result = "This email address does not match known scam domain patterns. Still verify the sender independently."
    return render_template('email_check.html', address_result=address_result)

@app.route('/number-check', methods=['GET', 'POST'])
def number_check():
    number_result = None
    if request.method == 'POST':
        number = request.form['number']
        number_result = analyze_phone_number(number)
    return render_template('number_check.html', number_result=number_result)

@app.route('/account-check', methods=['GET', 'POST'])
def account_check():
    account_result = None
    if request.method == 'POST':
        account = request.form['account'].strip()
        is_valid_account = account.isdigit() and 9 <= len(account) <= 18
        is_valid_upi = '@' in account and len(account.split('@')) == 2 and all(account.split('@'))
        if not (is_valid_account or is_valid_upi):
            account_result = "INVALID INPUT: Please enter a valid bank account number (9-18 digits) or a valid UPI ID (e.g., name@bank)."
        else:
            match = fraud_accounts_db[fraud_accounts_db['account_number'] == account]
            if not match.empty:
                category = match.iloc[0]['category']
                reports = match.iloc[0]['reports']
                account_result = f"WARNING: This account is reported as SCAM! Category: {category}, Reports: {reports}"
            else:
                account_result = "This account has no scam reports in our database. Still be cautious."
    return render_template('number_check.html', account_result=account_result)

if __name__ == '__main__':
    app.run(debug=True)
