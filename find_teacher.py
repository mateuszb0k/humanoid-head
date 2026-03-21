import json
import os

def get_teacher_room(teacher_name, db_filepath="teachers_info.json"):
    if not os.path.exists(db_filepath):
        return f"Error: Database file '{db_filepath}' not found."

    # JSON database
    with open(db_filepath, 'r', encoding='utf-8') as file:
        try:
            teachers_db = json.load(file)
        except json.JSONDecodeError:
            return f"Error: File '{db_filepath}' is not a valid JSON file."

    # Search
    if teacher_name in teachers_db:
        teacher_info = teachers_db[teacher_name]
        
        building = teacher_info.get("building", "N/A")
        room = teacher_info.get("room", "N/A")
        
        # Checking if room actually exists (handling "Brak")
        if room == "Brak" or room == "N/A":
            return f"Informacja: Nauczyciel '{teacher_name}' nie ma przypisanego pokoju w bazie."
        
        # Returning string formatted specifically for find_room.py
        return f"{building}, {room}"
        
    else:
        return f"Błąd: Nauczyciel '{teacher_name}' nie został znaleziony w bazie."

if __name__ == "__main__":
    target_person = "Name Surname"     # person we wanna find
    output_teacher = get_teacher_room(target_person)
    
    print(f"Wynik wyszukiwania: {output_teacher}")