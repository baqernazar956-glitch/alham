
import os

file_path = r"c:\Users\al6md\Desktop\project alham\flask_book_recommendation_starter\flask_book_recommendation\recommender.py"

try:
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Remove null bytes
    clean_content = content.replace(b'\x00', b'')
    
    # Remove UTF-16 BOM if present (created by PowerShell >> sometimes)
    if clean_content.startswith(b'\xff\xfe'):
        # This is UTF-16 LE
        decoded = clean_content.decode('utf-16-le')
        clean_content = decoded.encode('utf-8')
    elif clean_content.startswith(b'\xfe\xff'):
        # UTF-16 BE
        decoded = clean_content.decode('utf-16-be')
        clean_content = decoded.encode('utf-8')
        
    # Also double check for double-encoding or mixed encoding
    # Ensuring it writes back as UTF-8
    with open(file_path, 'wb') as f:
        f.write(clean_content)
        
    print("Fixed file successfully.")
except Exception as e:
    print(f"Error: {e}")
