import json

def get_room_directions(input_filepath, database_filepath):
    try:
        with open(database_filepath, 'r', encoding='utf-8') as db_file:
            database = json.load(db_file)
    except FileNotFoundError:
        return "Błąd: Nie znaleziono pliku"

    try:
        with open(input_filepath, 'r', encoding='utf-8') as in_file:
            search_request = json.load(in_file)
    except FileNotFoundError:
        return "Błąd: Nie znaleziono pliku wej."

    # extract the variables we want to search for
    target_building = search_request.get("building")
    target_room = search_request.get("number")

    # searching
    # building
    if target_building in database:
        building_data = database[target_building]
        # room
        if target_room in building_data:
            room_info = building_data[target_room]
            floor_number = room_info["floor"]
            raw_directions = room_info["directions"]
            final_directions = raw_directions.format(floor=floor_number)    # floor into the text
            
            return f"--- Wynik wyszukiwania ---\nBudynek: {target_building}\nPokój: {target_room}\nPiętro: {floor_number}\nWskazówki: {final_directions}"
        else:
            return f"Błąd: Pokój '{target_room}' nie istnieje w budynku '{target_building}'."
    else:
        return f"Błąd: Budynek '{target_building}' nie istnieje w bazie."

if __name__ == "__main__":
    result = get_room_directions('input.json', 'rooms.json')    
    print(result)