from flask import Flask, render_template, request
import pickle
import pandas as pd
import phonenumbers
from phonenumbers import geocoder
from PIL import Image
import pytesseract
import os

app = Flask(__name__)

# ---------- Load ML models safely ----------
def safe_load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"WARNING: could not load {path}: {e}")
        return None

message_model = safe_load_pickle('models/spam_model.pkl')
message_vectorizer = safe_load_pickle('models/vectorizer.pkl')
email_model = safe_load_pickle('models/email_model.pkl')
email_vectorizer = safe_load_pickle('models/email_vectorizer.pkl')

# ---------- Load datasets safely ----------
def safe_load_csv(path, columns):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"WARNING: could not load {path}: {e}")
        return pd.DataFrame(columns=columns)

scam_db = safe_load_csv('datasets/scam_numbers.csv', ['number', 'category', 'reports'])
scam_db['number'] = scam_db['number'].astype(str)

fraud_accounts_db = safe_load_csv('datasets/fraud_accounts.csv', ['account_number', 'category', 'reports'])
fraud_accounts_db['account_number'] = fraud_accounts_db['account_number'].astype(str)

special_db = safe_load_csv('datasets/special_numbers.csv', ['number', 'type', 'description'])
special_db['number'] = special_db['number'].astype(str)

# ---------- Keyword lists ----------
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
    "pay to confirm your seat", "processing charge",
    "sbi", "state bank of india", "bank of baroda", "bob", "hdfc bank",
    "icici bank", "axis bank", "pnb", "punjab national bank", "canara bank",
    "union bank", "kotak bank", "yes bank", "your account is suspended",
    "your account will be deactivated", "kyc verification pending",
    "update kyc immediately", "complete your kyc", "link your aadhaar",
    "electricity bill due", "your electricity connection will be disconnected",
    "power bill overdue", "pay electricity bill now", "bijli bill",
    "university fee payment", "scholarship approved", "admission confirmed pay fee",
    "income tax refund", "gst refund", "pan card update", "pan card blocked",
    "lottery winner", "you have won", "cash prize", "claim your reward",
    "loan approved instantly", "pre-approved loan", "credit card blocked",
    "debit card blocked", "atm card blocked", "your card will expire",
    "withdrawal request", "deposit confirmation pending", "transaction failed retry",
    "refund initiated click here", "netbanking suspended", "upi id blocked",
    "verify upi pin", "share upi pin", "share atm pin", "share cvv",
    "government scheme", "pm kisan yojana", "free recharge", "cashback offer expiring",
    "parcel held at customs", "courier delivery failed pay fee", "customs duty pending"
]

SUSPICIOUS_DOMAIN_PATTERNS = [
    "sbi-verify", "rbi-alert", "kbc-lottery", "bank-update",
    "verify-account", "secure-login", "-support", "kyc-update"
]

def check_red_flags(text):
    text_lower = text.lower()
    return [flag for flag in RED_FLAGS if flag in text_lower]

def get_flag_status(reports):
    if reports > 0:
        return "RED"
    return "GREEN"

# ---------- Phone number logic ----------
def check_indian_number(core):
    if core[0] not in ['6', '7', '8', '9']:
        return {
            "message": "INVALID: This is not a valid Indian mobile number. Indian mobile numbers must start with 6, 7, 8, or 9.",
            "flag": "RED", "reports": 0, "category": None,
            "series_type": "Invalid/Non-existent series", "number_length": len(core), "country": "India"
        }

    match = scam_db[scam_db['number'] == core]
    series_type = "Mobile Number"

    if not match.empty:
        category = match.iloc[0]['category']
        reports = int(match.iloc[0]['reports'])
        flag = get_flag_status(reports)
        message = f"WARNING: This is an Indian (+91) number reported as SCAM! Category: {category}, Reports: {reports}"
    else:
        category = None
        reports = 0
        flag = "GREEN"
        message = "This is a valid Indian (+91) number format with no scam reports in our database. Still be cautious."

    return {
        "message": message, "flag": flag, "reports": reports, "category": category,
        "series_type": series_type, "number_length": len(core), "country": "India"
    }

def analyze_phone_number(raw_number):
    cleaned = raw_number.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")

    if not cleaned:
        return {
            "message": "Please enter a phone number.", "flag": "RED", "reports": 0,
            "category": None, "series_type": "N/A", "number_length": 0, "country": "Unknown"
        }

    # Check special/emergency/helpline numbers first
    special_match = special_db[special_db['number'] == cleaned]
    if not special_match.empty:
        stype = special_match.iloc[0]['type']
        desc = special_match.iloc[0]['description']
        return {
            "message": f"{stype.upper()}: {cleaned} is a recognized {stype} number — {desc}.",
            "flag": "GREEN", "reports": 0, "category": None,
            "series_type": stype, "number_length": len(cleaned), "country": "India"
        }

    # Indian 10-digit or +91/91 numbers
    if cleaned.startswith("+91") and cleaned[3:].isdigit() and len(cleaned[3:]) == 10:
        return check_indian_number(cleaned[3:])
    if cleaned.startswith("91") and len(cleaned) == 12 and cleaned.isdigit():
        return check_indian_number(cleaned[2:])
    if cleaned.isdigit() and len(cleaned) == 10:
        return check_indian_number(cleaned)

    # International numbers via phonenumbers (public numbering-plan library)
    try:
        parsed = phonenumbers.parse(cleaned if cleaned.startswith("+") else "+" + cleaned, None)
        if phonenumbers.is_valid_number(parsed):
            country = geocoder.description_for_number(parsed, "en") or "Unknown region"
            return {
                "message": f"This is a valid international number registered under {country}.",
                "flag": "GREEN", "reports": 0, "category": None,
                "series_type": "International", "number_length": len(cleaned), "country": country
            }
        else:
            return {
                "message": "INVALID: This does not match any known valid phone number format worldwide.",
                "flag": "RED", "reports": 0, "category": None,
                "series_type": "N/A", "number_length": len(cleaned), "country": "Unknown"
            }
    except Exception:
        return {
            "message": "INVALID INPUT: Please enter a valid phone number (India or International, with country code).",
            "flag": "RED", "reports": 0, "category": None,
            "series_type": "N/A", "number_length": len(cleaned), "country": "Unknown"
        }

# ---------- Routes ----------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/number-check', methods=['GET', 'POST'])
def number_check():
    number_result = None
    submitted_number = ''
    if request.method == 'POST':
        number = request.form.get('number', '').strip()
        submitted_number = number
        number_result = analyze_phone_number(number)
    return render_template('number_check.html', number_result=number_result, submitted_number=submitted_number)

@app.route('/account-check', methods=['GET', 'POST'])
def account_check():
    account_result = None
    submitted_account = ''
    if request.method == 'POST':
        account = request.form.get('account', '').strip()
        submitted_account = account
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
    return render_template('number_check.html', account_result=account_result, submitted_account=submitted_account)

@app.route('/message-check', methods=['GET', 'POST'])
def message_check():
    result = None
    submitted_message = ''
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        submitted_message = message
        if len(message) < 15 or message.replace(" ", "").isdigit():
            result = "This does not look like a real message. Please paste the full SMS/WhatsApp/email text for proper scanning."
        else:
            flags = check_red_flags(message)
            ml_prediction = 0
            if message_model is not None and message_vectorizer is not None:
                try:
                    vec = message_vectorizer.transform([message])
                    ml_prediction = message_model.predict(vec)[0]
                except Exception:
                    ml_prediction = 0
            if ml_prediction == 1 or len(flags) > 0:
                result = "RED — This looks like a SCAM message! Be careful."
                if flags:
                    result += " (Red flags found: " + ", ".join(flags) + ")"
            else:
                result = "GREEN — This message looks SAFE. No known fraud patterns detected."
    return render_template('message_check.html', result=result, submitted_message=submitted_message)

@app.route('/email-check', methods=['GET', 'POST'])
def email_check():
    result = None
    submitted_email_text = ''
    if request.method == 'POST':
        email_text = request.form.get('email_text', '').strip()
        submitted_email_text = email_text
        if len(email_text) < 15 or email_text.replace(" ", "").isdigit():
            result = "This does not look like a real email. Please paste the full email text for proper scanning."
        else:
            flags = check_red_flags(email_text)
            ml_prediction = 0
            if email_model is not None and email_vectorizer is not None:
                try:
                    vec = email_vectorizer.transform([email_text])
                    ml_prediction = email_model.predict(vec)[0]
                except Exception:
                    ml_prediction = 0
            if ml_prediction == 1 or len(flags) > 0:
                result = "RED — This looks like a FAKE/SCAM email! Be careful."
                if flags:
                    result += " (Red flags found: " + ", ".join(flags) + ")"
            else:
                result = "GREEN — This email looks GENUINE."
    return render_template('email_check.html', result=result, submitted_email_text=submitted_email_text)

@app.route('/email-check-image', methods=['POST'])
def email_check_image():
    image_result = None
    try:
        file = request.files.get('email_image')
        if file:
            img = Image.open(file)
            extracted_text = pytesseract.image_to_string(img)
            if extracted_text.strip():
                flags = check_red_flags(extracted_text)
                ml_prediction = 0
                if email_model is not None and email_vectorizer is not None:
                    try:
                        vec = email_vectorizer.transform([extracted_text])
                        ml_prediction = email_model.predict(vec)[0]
                    except Exception:
                        ml_prediction = 0
                if ml_prediction == 1 or len(flags) > 0:
                    image_result = "RED — This looks like a FAKE/SCAM email! Be careful."
                    if flags:
                        image_result += " (Red flags found: " + ", ".join(flags) + ")"
                else:
                    image_result = "GREEN — This email looks GENUINE."
            else:
                image_result = "Could not read text from the image. Try a clearer screenshot."
        else:
            image_result = "Please upload an image file."
    except Exception as e:
        image_result = "Something went wrong while reading the image. Please try a different file."
    return render_template('email_check.html', image_result=image_result)

@app.route('/email-address-check', methods=['GET', 'POST'])
def email_address_check():
    address_result = None
    submitted_sender_email = ''
    if request.method == 'POST':
        sender_email = request.form.get('sender_email', '').strip().lower()
        submitted_sender_email = sender_email
        if not sender_email or '@' not in sender_email:
            address_result = "INVALID INPUT: Please enter a valid email address."
        else:
            domain = sender_email.split('@')[-1]
            found = [pattern for pattern in SUSPICIOUS_DOMAIN_PATTERNS if pattern in domain]
            if found:
                address_result = f"RED — SUSPICIOUS: Domain matches known fake-bank/scam patterns: {', '.join(found)}"
            else:
                address_result = "GREEN — This email address does not match known scam domain patterns. Still verify the sender independently."
    return render_template('email_check.html', address_result=address_result, submitted_sender_email=submitted_sender_email)

# ---------- Error handlers (so users never see raw crash pages) ----------
@app.errorhandler(404)
def not_found_error(error):
    return "Page not found. Please go back and try again.", 404

@app.errorhandler(500)
def internal_error(error):
    return "Something went wrong on our end. Please try again in a moment.", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
