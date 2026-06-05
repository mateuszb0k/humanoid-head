import sqlite3
import pickle

def view_database():
    try:
        with sqlite3.connect("faces.db") as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT id, name, encoding FROM users")
            users = cursor.fetchall()
            if not users:
                print("Database is empty")
                return
            print(f"{'ID':<5} | {'Name':<15} | Encoding data")
            print("-" * 55)
            for user_id, name, binary_data in users:
                try:
                    vec = pickle.loads(binary_data)
                    vector_info = f"[Array saved: {len(vec)} features]"
                except Exception:
                    vector_info = "[Corrupted data]"
                print(f"{user_id:<5} | {name:<15} | {vector_info}")
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    view_database()