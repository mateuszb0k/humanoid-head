import json
import os
from rapidfuzz import process,fuzz
THRESHOLD = 75 #% seems to be optimal
def get_teacher_room(teacher_name, db_filepath="teachers_info.json"):
    empty_result = {
        "teacher_name": None,
        "room": None,
        "building": None,
    }
    if not os.path.exists(db_filepath):
        print( f"Error: Database file '{db_filepath}' not found.")
        return empty_result


    # fetch data
    with open(db_filepath, 'r', encoding='utf-8') as file:
        try:
            teachers_db = json.load(file)
        except json.JSONDecodeError:
            print(f"Error: File '{db_filepath}' is not a valid JSON file.")
            return empty_result

    # Search w/ fuzzy matching
    keys = teachers_db.keys()
    key,score,_ = process.extractOne(teacher_name, keys, scorer= fuzz.WRatio) #fuzzy matching
    print(key,score)
    if score>=THRESHOLD:
        teacher_info  = teachers_db[key]
        room = teacher_info.get("room",None)
        building = teacher_info.get("building",None)
        if room!="Brak" and building!="Brak":
            result = {
                "teacher_name": key,
                "room": room,
                "building": building,
            }
            return result
        else:
            #case where no room is found for a valid key
            result = {
                "teacher_name": key,
                "room": None,
                "building": None,
            }
            return result
    else:
        return empty_result


if __name__ == "__main__":
    target_person = "mgr dr hab prof inz Place Holder"  # person to be found
    output_teacher = get_teacher_room(target_person)

    print(f"result: {output_teacher}")