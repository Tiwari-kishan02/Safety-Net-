from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "SafetyNet+ backend is running!"

@app.route('/recovery')
def recovery():
    return render_template('recovery.html')

if __name__ == '__main__':
    app.run(debug=True)
