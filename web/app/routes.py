from flask import Flask, render_template, redirect
# from app import app
from deckgen import Deckgen

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deckgen')
def deckgen():
    gen = Deckgen()
    return redirect(gen.sample())

if __name__ == '__main__':
    from os import environ
    app.run(debug=False, port=environ.get("PORT", 5000), host='0.0.0.0')