
from flask_book_recommendation.app import create_app, db

print("Initializing database...")
app = create_app()

with app.app_context():
    try:
        db.create_all()
        print("db.create_all() executed successfully.")
        
        # Verify
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        print("Tables created:", inspector.get_table_names())
        
    except Exception as e:
        print(f"Error creating tables: {e}")
