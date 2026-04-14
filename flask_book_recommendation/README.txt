Book Recommendation System (Flask) - EN
======================================

Quick Start
-----------
1) Create DB in MySQL:
   CREATE DATABASE book_recommendation;

2) Update config.py if your MySQL password differs.

3) Create venv & install deps:
   py -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

4) Run the app:
   py app.py
   Open http://127.0.0.1:5000/

Seed Demo Books (optional)
--------------------------
Open once: http://127.0.0.1:5000/seed/books
Then visit: http://127.0.0.1:5000/books

Project Structure
-----------------
- app.py, config.py, extensions.py, models.py
- routes/: auth.py, main.py
- templates/: base.html, home.html, login.html, register.html, books.html

Notes
-----
- Users table stores password hashes (Werkzeug).
- Change SECRET_KEY in config.py for production.
