import json

def get_aud_directions(input_room_str, database_filepath='./db_specialrooms.json'):
    """
    Retrieves navigation directions for specialized rooms (e.g., auditoriums, libraries).

    Parses the normalized input string and queries the JSON database
    to find the corresponding routing instructions.

    Args:
        A normalized string containing the building prefix and 
            the room name separated by a comma (e.g., "NE, AUD1 LEWE").
        Path to the database file.

    Returns:
        str: The explicit navigation instructions as text.
    """
    #print(input_room_str)
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
    # local test
    input_room = "NE, AUD1 LEWE"
    output_room = get_aud_directions(input_room)    
    print(output_room)