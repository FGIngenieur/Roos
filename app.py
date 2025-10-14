from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pathlib import Path
import json
from datetime import datetime

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
MESSAGES_FILE = DATA_DIR / "messages.json"
if not MESSAGES_FILE.exists():
    MESSAGES_FILE.write_text("[]", encoding="utf-8")

app = Flask(__name__)
app.secret_key = "change_this_secret_in_production"

def read_messages():
    try:
        return json.loads(MESSAGES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_message(msg):
    messages = read_messages()
    messages.append(msg)
    MESSAGES_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

@app.route("/")
def index():
    return render_template("index.html", title="Accueil")

@app.route("/about")
def about():
    return render_template("about.html", title="À propos")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not message:
            flash("Veuillez remplir tous les champs.", "danger")
            return redirect(url_for("contact"))

        entry = {
            "name": name,
            "email": email,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        save_message(entry)
        flash("Merci ! Ton message a bien été enregistré.", "success")
        return redirect(url_for("contact"))

    return render_template("contact.html", title="Contact")

# API utile pour récupérer les messages (ex : usage admin)
@app.route("/api/messages", methods=["GET"])
def api_messages():
    # NOTE: en production, protége avec auth !
    return jsonify(read_messages())

if __name__ == "__main__":
    app.run(debug=True, port=5000)
