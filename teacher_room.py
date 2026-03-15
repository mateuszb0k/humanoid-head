import json
import os

def get_teacher_room(teacher_name, db_filepath="teachers.json", output_filepath="teacher_room.json"):
    if not os.path.exists(db_filepath):
        print(f"Error: Database file '{db_filepath}' not found.")
        return

    # JSON database
    with open(db_filepath, 'r', encoding='utf-8') as file:
        try:
            teachers_db = json.load(file)
        except json.JSONDecodeError:
            print(f"Error: File '{db_filepath}' is not a valid JSON file.")
            return

    # Search
    if teacher_name in teachers_db:
        teacher_info = teachers_db[teacher_name]
        
        result_data = {
            "building": teacher_info.get("building", "N/A"),
            "number": teacher_info.get("room", "N/A")
        }
        
        # output JSON file (for now)
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(result_data, outfile, indent=4, ensure_ascii=False)
            
        print(f"Data has been saved to the file: '{output_filepath}'")
        
    else:
        print(f"Info: Teacher '{teacher_name}' was not found in the database")

if __name__ == "__main__":
    target_person = "Name Surname"     # person we wanna find
    get_teacher_room(target_person)