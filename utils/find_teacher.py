import json
import os
from rapidfuzz import process,fuzz
THRESHOLD = 50
TOP_3_THRESHOLD = 90
#% seems to be optimal
def get_teacher_room(teacher_name: str, db_filepath="./teachers_info.json"):
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

    #Search w/ fuzzy matching
    teacher_info = teachers_db[teacher_name]
    room = teacher_info['room']
    building = teacher_info['building']
    if room!="Brak" and building!="Brak":
            result = {
                "teacher_name": teacher_name,
                "room": room,
                "building": building,
            }
            return result
    else:
            #case where no room is found for a valid key
            result = {
                "teacher_name": teacher_name,
                "room": None,
                "building": None,
            }
            return result
#used to get top n results
def search_teacher(teacher_name: str, db_filepath="./teachers_info.json",top_n = 3):
    teacher_dict = {}
    if not os.path.exists(db_filepath):
        print( f"Error: Database file '{db_filepath}' not found.")
        return teacher_dict
    # fetch data
    with open(db_filepath, 'r', encoding='utf-8') as file:
        try:
            teachers_db = json.load(file)
        except json.JSONDecodeError:
            print(f"Error: File '{db_filepath}' is not a valid JSON file.")
            return teacher_dict
    keys = teachers_db.keys()
    l = process.extract(teacher_name, keys, scorer=fuzz.WRatio, limit=top_n)
    for el in l:
        teacher_dict[el[0]] = el[1]
    return teacher_dict

if __name__ == "__main__":
    top_3 = search_teacher("Michał ") #person to be found
    best_val = max(top_3.values())
    best_teacher = None
    for k, v in top_3.items():
        if v == best_val:
            best_teacher = k
    if best_val >TOP_3_THRESHOLD:
        full_data = get_teacher_room(best_teacher)
        print(full_data)
    elif best_val > THRESHOLD:
        for k,v in top_3.items():
            print(f"{k}: {v}")
    else:
        print("No teacher found")
    # print(f"result: {output_teacher}")