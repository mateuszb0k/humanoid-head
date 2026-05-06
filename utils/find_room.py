import json

def get_room_directions(input_room_str, database_filepath='./room_directions.json'):
    print(input_room_str)
    # reading database
    try:
        with open(database_filepath, 'r', encoding='utf-8') as db_file:
            database = json.load(db_file)
    except FileNotFoundError:
        return "Error: data not found"

    # "tokening"
    try:
        parts = input_room_str.split(',')
        target_building = parts[0].strip()
        target_room = parts[1].strip()
    except IndexError:
        return "Error: wrong form"

    # finding
    if target_building in database:
        building_data = database[target_building]
        
        if target_room in building_data:
            return building_data[target_room]["directions"]
        else:
            return f"Błąd Pokój '{target_room}' nie istnieje w budynku '{target_building}'."
    else:
        return f"Błąd Budynek '{target_building}' nie istnieje w bazie. "

if __name__ == "__main__":
    # input as string
    input_room = "EA, 107"
    # output as string
    output_room = get_room_directions(input_room)    
    print(output_room)