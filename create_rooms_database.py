import json

# Initialize database 
building_db = {
    "EA": {},
    "NE": {
        "105": {"floor": "1", "directions": "Pójdź w lewo od głównego wejścia, po schodach lub windą udaj się na {floor} piętro i wejdź w korytarz po lewej stronie"},
        "205": {"floor": "2", "directions": "Pójdź w lewo od głównego wejścia, po schodach lub windą udaj się na {floor} piętro i wejdź w korytarz po lewej stronie"},
        "215": {"floor": "2", "directions": "Pójdź na prawo od głównego wejścia, po schodach lub windą udaj się na {floor} piętro i wejdź w korytarz po prawej stronie"}
    }
}

def get_floor(room_num):
    return room_num // 100

# Middle rooms (Floors 1-7)
mid_rooms = [126, 130, 222, 224, 328, 330, 436, 438, 532, 534, 628, 630, 734, 736]
mid_template = "Udaj się na piętro {floor} środkową windą lub schodami. sala {room} będzie się znajdować na środku holu, na przeciwko okien."

for room in mid_rooms:
    floor = get_floor(room)
    building_db["EA"][str(room)] = {
        "floor": str(floor),
        "directions": mid_template.format(floor=floor, room=room)
    }

# 2. Upper floors rules (Floors 1-7)
upper_rules = [
    # Left, left side (Odd)
    (
        [(137, 149), (237, 255), (339, 353), (441, 451), (539, 551), (641, 651), (741, 753)],
        "odd",
        "Pojedź windą lub pójdź schodami na piętro {floor} i pójdź holem w lewo, sala {room} będzie znajdowała się po lewej stronie holu"
    ),
    # Left, right side (Even)
    (
        [(134, 148), (226, 242), (334, 350), (440, 456), (536, 552), (632, 648), (738, 754)],
        "even",
        "Pojedź windą lub pójdź schodami na piętro {floor} i pójdź holem w lewo, sala {room} będzie znajdowała się po prawej stronie holu"
    ),
    # Right, left side (Even)
    (
        [(102, 124), (202, 220), (302, 328), (402, 434), (502, 530), (600, 626), (702, 732)],
        "even",
        "Pojedź windą lub pójdź schodami na piętro {floor} i pójdź holem w prawo, sala {room} będzie znajdowała się po lewej stronie holu"
    ),
    # Right, right side (Odd)
    (
        [(99, 133), (201, 235), (301, 331), (401, 433), (501, 531), (601, 633), (701, 735)],
        "odd",
        "Pojedź windą lub pójdź schodami na piętro {floor} i pójdź holem w prawo, sala {room} będzie znajdowała się po prawej stronie holu"
    )
]

# 3. Ground floor rules (Floor 0)
floor0_rules = [
    # Right, left side (Odd)
    (
        [(55, 69)],
        "odd",
        "Wejdź do budynku głównym wejściem i pójdź holem w prawo, sala {room} będzie znajdowała się po lewej stronie holu"
    ),
    # Right, right side (Even)
    (
        [(26, 38)],
        "even",
        "Wejdź do budynku głównym wejściem i pójdź holem w prawo, sala {room} będzie znajdowała się po prawej stronie holu"
    ),
    # Left, left side (Even)
    (
        [(2, 24)],
        "even",
        "Wejdź do budynku głównym wejściem i pójdź holem w lewo, sala {room} będzie znajdowała się po lewej stronie holu"
    ),
    # Left, right side (Odd)
    (
        [(1, 31)],
        "odd",
        "Wejdź do budynku głównym wejściem i pójdź holem w lewo, sala {room} będzie znajdowała się po prawej stronie holu"
    )
]

# Process both sets of rules (Upper floors and Floor 0)
all_rules = upper_rules + floor0_rules

for ranges, parity, template in all_rules:
    for start, end in ranges:
        for room in range(start, end + 1):
            # Check parity (even or odd)
            if (parity == "odd" and room % 2 != 0) or (parity == "even" and room % 2 == 0):
                floor = get_floor(room)
                if str(room) not in building_db["EA"]:
                    building_db["EA"][str(room)] = {
                        "floor": str(floor),
                        "directions": template.format(floor=floor, room=room)
                    }
 
building_db["EA"]["AUD.1"] = {
    "floor": "0",
    "directions": "Wejdź do budynku głównym wejściem, pójdź na wprost, wejdź po schodkach, audytorium będzie znajdowało się po lewej stronie"
}

building_db["EA"]["AUD.2"] = {
    "floor": "0",
    "directions": "Wejdź do budynku głównym wejściem, pójdź na wprost, wejdź po schodkach, audytorium będzie znajdowało się po prawej stronie"
}

building_db["EA"]["szatnia"] = {
    "floor": "-1",
    "directions": "Szatnia znajduje się na -1, wystarczy, że zejdziesz schodami lub zjedziesz windą i na środku znajdziesz szatnię, gdzie możesz powiesić swoją kurtkę"
}

# Numbers first , then strings (AUD.1, szatnia)
def custom_sort(item):
    key = item[0]
    if key.isdigit():
        return (0, int(key))
    return (1, key)

building_db["EA"] = dict(sorted(building_db["EA"].items(), key=custom_sort))

# Save to JSON file
with open("room_directions.json", "w", encoding="utf-8") as f:
    json.dump(building_db, f, ensure_ascii=False, indent=2)

print("Success! The updated database has been generated and saved to 'room_directions.json'.")