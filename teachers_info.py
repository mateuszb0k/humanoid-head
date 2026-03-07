import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin

def scrape_smart_pg_teachers(main_url):
    print(f"Krok 1: Wchodzę na {main_url} i szukam ukrytych linków do profili...")
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(main_url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Błąd połączenia: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    profile_links = set()
    
    # Pobieranie linków
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if '/p/' in href: 
            full_link = urljoin(main_url, href)
            profile_links.add(full_link)

    print(f"Znaleziono {len(profile_links)} profili. Zaczynam inteligentne skanowanie!\n")
    
    teachers_database = {}

    for link in profile_links:
        try:
            print(f"Skanuję: {link}")
            prof_resp = requests.get(link, headers=headers)
            prof_soup = BeautifulSoup(prof_resp.text, 'html.parser')
            
            # imie nazwisko tytul
            raw_name_tag = prof_soup.find('h1')
            if not raw_name_tag:
                continue
            
            raw_name = raw_name_tag.get_text(strip=True).replace(" | Politechnika Gdańska", "")
            
            words = raw_name.split()
            title_parts = []
            name_parts = []
            
            for word in words:
                if word[0].isupper():
                    name_parts.append(word)
                else:
                    title_parts.append(word)
                    
            title = " ".join(title_parts) if title_parts else "Brak"
            clean_name = " ".join(name_parts)
            
            raw_text = prof_soup.get_text(separator=' ', strip=True) # Do szukania pokoju i emaila
            raw_text_piped = prof_soup.get_text(separator='|', strip=True) # Do szukania konkretnych sekcji
            
            # email
            email = "Brak"
            email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', raw_text)
            if email_match:
                email = email_match.group(1).strip()
                
            # katedra
            department = "Brak"
            # Szukamy tekstu "miejsce pracy:|", a następnie łapiemy wszystko do kolejnego znaku "|"
            dept_match = re.search(r'miejsce pracy:\|([^|]+)', raw_text_piped, re.IGNORECASE)
            if dept_match:
                department = dept_match.group(1).strip()
            
            # budynek i pokoj
            building = "EA" # Ustawiamy EA jako domyślne dla całego wydziału, jeśli nie ma napisane inaczej
            room_number = "Brak"
            
            explicit_match = re.search(r'(EA|NE)\s*(\d{2,4}[a-zA-Z]?)', raw_text)
            if explicit_match:
                building = explicit_match.group(1)
                room_number = explicit_match.group(2)
            else:
                implicit_match = re.search(r'(?:pok\.|p\.:|pokój)\s*(\d{2,4}[a-zA-Z]?)', raw_text, re.IGNORECASE)
                if implicit_match:
                    room_number = implicit_match.group(1)

            # zapis
            teachers_database[clean_name] = {
                "title": title,
                "building": building,
                "room": room_number,
                "email": email,
                "department": department
            }
            
        except Exception as e:
            print(f"Nie udało się pobrać danych z {link}: {e}")
            continue

    output_filename = 'teachers_details.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(teachers_database, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Gotowe! Pomyślnie sformatowano bazę z {len(teachers_database)} pracownikami.")

if __name__ == "__main__":
    TARGET_URL = "https://eti.pg.edu.pl/KIBM/pracownicy-katedry"    
    # katedry: kams kask KIBM ksdr <- done   kima kiop kisi  kmse kimf ksis ksgi ktin ksme ksmm kssr ksti koel
    scrape_smart_pg_teachers(TARGET_URL)