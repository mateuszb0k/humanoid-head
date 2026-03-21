import json

building_db = {
    "EA": {},
    "NE": {}
}

def get_floor(room_num):
    return room_num // 100

# RULES FOR BUILDING EA
# Middle rooms 
mid_rooms = [126, 130, 222, 224, 328, 330, 436, 438, 532, 534, 628, 630, 734, 736]
mid_template = "Udaj się na piętro {floor} środkową windą lub schodami. sala {room} będzie się znajdować na środku holu, na przeciwko okien."

for room in mid_rooms:
    floor = get_floor(room)
    building_db["EA"][str(room)] = {
        "floor": str(floor),
        "directions": mid_template.format(floor=floor, room=room)
    }

# Upper floors rules
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

# Ground floor rules (Floor 0)
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

# Process both sets of rules for EA
all_rules_ea = upper_rules + floor0_rules

for ranges, parity, template in all_rules_ea:
    for start, end in ranges:
        for room in range(start, end + 1):
            if (parity == "odd" and room % 2 != 0) or (parity == "even" and room % 2 == 0):
                floor = get_floor(room)
                if str(room) not in building_db["EA"]:
                    building_db["EA"][str(room)] = {
                        "floor": str(floor),
                        "directions": template.format(floor=floor, room=room)
                    }

# Special rooms EA
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


# RULES FOR BUILDING NE
# Special rooms on floor 0 and 1
special_rooms_ne = {
    "Szatnia": {"floor": "0", "directions": "W budynku Nowego Eti znajdują się dwie szatnie, wejdź głównym wejściem przejdź przez dwoje drzwi po prawej i zobaczysz szatnię lub wejdź głównym wejściem przejdź przez dwoje drzwi po lewej i tam zobaczysz szatnię."},
    "Stołówka": {"floor": "0", "directions": "Wejdź głównym wejściem, pójdź na wprost lekko w lewo, tam będziesz mógł coś zjeść i się napić."},
    "Aud 2": {"floor": "0", "directions": "Wejdź głównym wejściem przejdź przez dwoje drzwi po lewej, następnie okrąż szatnię z lewej lub z prawej strony."},
    "Biblioteka": {"floor": "0", "directions": "Wejdź głównym wejściem przejdź przez dwoje drzwi po lewej, następnie omiń szatnię i po prawej stronie znajdziesz bibliotekę."},
    "51": {"floor": "0", "directions": "Wejdź głównym wejściem przejdź przez dwoje drzwi po lewej, następnie omiń szatnię i po prawej stronie znajdziesz bibliotekę i tam na wprost wejścia znajdziesz salę 51."},
    "Aud1L": {"floor": "1", "directions": "Wejdź do budynku głównym wejściem skieruj się na schody na wprost od prawej strony, wejdź na pierwsze piętro i sala wykładowa będzie znajdować się po prawej stronie."},
    "Aud1P": {"floor": "1", "directions": "Wejdź do budynku głównym wejściem skieruj się na schody na wprost od lewej strony, wejdź na pierwsze piętro i sala wykładowa będzie znajdować się po lewej stronie."}
}

for room_name, data in special_rooms_ne.items():
    building_db["NE"][room_name] = data

# Numeric rules for NE
numeric_rules_ne = [
    # 104, 105, 106, 204, 205, 206
    (
        [104, 105, 106, 204, 205, 206],
        "Wejdź głównym wejściem przejdź przez drzwi po lewej stronie, następnie udaj się na poziom {floor}, wejdź do holu po lewej stronie i sala {room} będzie znajdować się po lewej stronie."
    ),
    # 109, 110, 207, 208, 209
    (
        [109, 110, 207, 208, 209],
        "Wejdź głównym wejściem przejdź przez drzwi po lewej stronie, następnie udaj się na poziom {floor}, wejdź do holu po lewej stronie i sala {room} będzie znajdować się po prawej stronie."
    ),
    # 140-147, 230, 231, 232 (using list extension for the range)
    (
        list(range(140, 148)) + [230, 231, 232],
        "Wejdź głównym wejściem przejdź przez drzwi po prawej stronie, następnie udaj się na poziom {floor}, wejdź do holu po prawej stronie i sala {room} będzie znajdować się po lewej stronie."
    ),
    # 149, 151, 153, 155, 156, 157, 233, 234, 235
    (
        [149, 151, 153, 155, 156, 157, 233, 234, 235],
        "Wejdź głównym wejściem przejdź przez drzwi po prawej stronie, następnie udaj się na poziom {floor}, wejdź do holu po prawej stronie i sala {room} będzie znajdować się po prawej stronie."
    ),
    # 159
    (
        [159],
        "Wejdź głównym wejściem przejdź przez drzwi po lewej stronie, następnie udaj się na poziom {floor}, wejdź do holu po prawej stronie i skręć w prawo, sala {room} będzie znajdować się po lewej stronie."
    ),
    # 160, 238
    (
        [160, 238],
        "Wejdź głównym wejściem przejdź przez drzwi po lewej stronie, następnie udaj się na poziom {floor}, wejdź do holu po prawej stronie i skręć w lewo, sala {room} będzie znajdować się po prawej stronie."
    ),
    # 161, 239
    (
        [161, 239],
        "Wejdź głównym wejściem przejdź przez drzwi po prawej stronie, następnie udaj się na poziom {floor}, wejdź do holu po lewej stronie i skręć w prawo, sala {room} będzie znajdować się po lewej stronie."
    ),
    # 162, 237
    (
        [162, 237],
        "Wejdź głównym wejściem przejdź przez drzwi po prawej stronie, następnie udaj się na poziom {floor}, wejdź do holu po lewej stronie i skręć w lewo, sala {room} będzie znajdować się po prawej stronie."
    ),
    # 215, 216, 217, 323, 324, 325
    (
        [215, 216, 217, 323, 324, 325],
        "Wejdź głównym wejściem przejdź przez drzwi po lewej stronie, następnie udaj się na poziom {floor}, wejdź do holu po prawej stronie i skręć w lewo, sala {room} będzie znajdować się na wprost."
    ),
    # 219, 220, 326, 327, 328
    (
        [219, 220, 326, 327, 328],
        "Wejdź głównym wejściem przejdź przez drzwi po prawej stronie, następnie udaj się na poziom {floor}, wejdź do holu po lewej stronie i skręć w prawo, sala {room} będzie znajdować się na wprost."
    ),
    # 309
    (
        [309],
        "Wejdź głównym wejściem przejdź przez drzwi po lewej stronie, następnie udaj się na poziom {floor}, wejdź do holu po lewej stronie sala {room} będzie znajdować się na wprost."
    ),
    # 310
    (
        [310],
        "Wejdź głównym wejściem przejdź przez drzwi po lewej stronie, następnie udaj się na poziom {floor}, wejdź do holu po lewej stronie, skręć w prawo i w lewo sala {room} będzie znajdować się na wprost."
    ),
    # 311-314
    (
        list(range(311, 315)),
        "Wejdź głównym wejściem przejdź przez drzwi po lewej stronie, następnie udaj się na poziom {floor}, wejdź do holu po lewej stronie, skręć w prawo i następnie w lewo sala {room} będzie znajdować się po prawej stronie korytarza."
    ),
    # 302-307
    (
        list(range(302, 308)),
        "Wejdź głównym wejściem przejdź przez drzwi po lewej stronie, następnie udaj się na poziom {floor}, wejdź do holu po lewej stronie, skręć w lewo i następnie w prawo sala {room} będzie znajdować się po lewej stronie korytarza."
    ),
    # 330, 332, 334, 335
    (
        [330, 332, 334, 335],
        "Wejdź głównym wejściem przejdź przez drzwi po prawej stronie, następnie udaj się na poziom {floor}, wejdź do holu po prawej stronie, skręć w lewo i następnie w prawo sala {room} będzie znajdować się po lewej stronie korytarza."
    ),
    # 343, 342, 339, 338, 336
    (
        [343, 342, 339, 338, 336],
        "Wejdź głównym wejściem przejdź przez drzwi po prawej stronie, następnie udaj się na poziom {floor}, wejdź do holu po prawej stronie, skręć w prawo i następnie w lewo sala {room} będzie znajdować się po prawej stronie korytarza."
    ),
    # 346
    (
        [346],
        "Wejdź głównym wejściem przejdź przez drzwi po prawej stronie, następnie udaj się na poziom {floor}, wejdź do holu po prawej stronie sala {room} będzie znajdować się na wprost."
    )
]

for room_list, template in numeric_rules_ne:
    for room in room_list:
        floor = get_floor(room)
        building_db["NE"][str(room)] = {
            "floor": str(floor),
            "directions": template.format(floor=floor, room=room)
        }

# Numbers first, then strings (AUD.1, szatnia)
def custom_sort(item):
    key = item[0]
    if key.isdigit():
        return (0, int(key))
    return (1, key)

# Sort both buildings
building_db["EA"] = dict(sorted(building_db["EA"].items(), key=custom_sort))
building_db["NE"] = dict(sorted(building_db["NE"].items(), key=custom_sort))

# Save to JSON file
with open("room_directions.json", "w", encoding="utf-8") as f:
    json.dump(building_db, f, ensure_ascii=False, indent=2)

print("Success! Database has been saved to 'room_directions.json'.")