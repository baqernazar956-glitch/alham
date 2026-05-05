import sqlite3
conn = sqlite3.connect('instance/app.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print("Tables:", tables)
for t in tables:
    cursor.execute(f"SELECT COUNT(*) FROM [{t}]")
    print(f"  {t}: {cursor.fetchone()[0]} rows")
# Show book columns
cursor.execute("PRAGMA table_info(book)")
print("\nBook columns:", [r[1] for r in cursor.fetchall()])
# Sample book
cursor.execute("SELECT google_books_id, title, author, genre FROM book LIMIT 3")
for r in cursor.fetchall():
    print(f"  Sample: gid={r[0]}, title={r[1]}, author={r[2]}, genre={r[3]}")
conn.close()
