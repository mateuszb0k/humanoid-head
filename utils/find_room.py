import json

import json
import re

def get_room_directions(input_room_str, database_filepath='./room_directions.json'):
    try:
        with open(database_filepath, 'r', encoding='utf-8') as db_file:
            database = json.load(db_file)
    except FileNotFoundError:
        return "Error: data not found"

    try:
        normalized = input_room_str.strip().upper()
        normalized = re.sub(r'[\.\s]+', '', normalized)
        normalized = re.sub(r'(?i)\b(NE|EA)(\d+)\b', r'\1,\2', normalized)

        parts = normalized.split(',')
        target_building = parts[0].strip()
        target_room = parts[1].strip()
    except IndexError:
        return "Error: wrong form"

    if target_building in database:
        building_data = database[target_building]

        if target_room in building_data:
            return building_data[target_room]["directions"]
        else:
            return f"Błąd Pokój '{target_room}' nie istnieje w budynku '{target_building}'."
    else:
        return f"Błąd Budynek '{target_building}' nie istnieje w bazie."
if __name__ == "__main__":
    # input as string
    input_room = "EA, 107"
    # output as string
    output_room = get_room_directions(input_room)    
    print(output_room)