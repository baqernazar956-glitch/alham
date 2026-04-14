import sqlite3
from werkzeug.security import generate_password_hash

conn = sqlite3.connect('flask_book_recommendation/app.db')
cursor = conn.cursor()

email = 'almagd1020@gmail.com'
password = '123456'  # You can change this

# Check if user already exists
cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
if cursor.fetchone():
    print(f"User {email} already exists!")
else:
    cursor.execute(
        'INSERT INTO users (name, email, password_hash, onboarding_completed) VALUES (?, ?, ?, ?)',
        ('Almagd', email, generate_password_hash(password), 0)
    )
    conn.commit()
    print(f"✓ User created successfully!")
    print(f"  Email: {email}")
    print(f"  Password: {password}")

conn.close()
