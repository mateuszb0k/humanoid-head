import requests
'''
Gets current weather in gdańsk
'''
WMO_CODES = {
    0: "Bezchmurnie",
    1: "Przeważnie bezchmurnie",
    2: "Częściowe zachmurzenie",
    3: "Zachmurzenie całkowite",
    45: "Mgła",
    48: "Mgła szronowa",
    51: "Mżawka lekka",
    53: "Mżawka umiarkowana",
    55: "Mżawka gęsta",
    56: "Marznąca mżawka lekka",
    57: "Marznąca mżawka gęsta",
    61: "Deszcz lekki",
    63: "Deszcz umiarkowany",
    65: "Deszcz ulewny",
    66: "Marznący deszcz lekki",
    67: "Marznący deszcz ulewny",
    71: "Śnieg lekki",
    73: "Śnieg umiarkowany",
    75: "Śnieg intensywny",
    77: "Ziarna śniegu",
    80: "Przelotny deszcz lekki",
    81: "Przelotny deszcz umiarkowany",
    82: "Przelotny deszcz gwałtowny",
    85: "Przelotne opady śniegu lekkie",
    86: "Przelotne opady śniegu intensywne",
    95: "Burza",
    96: "Burza z lekkim gradem",
    99: "Burza z intensywnym gradem",
}
def get_weather() -> dict:
    weather = {}
    try:
        data = requests.get('https://api.open-meteo.com/v1/forecast?latitude=54.3523&longitude=18.6491&hourly=uv_index&models=best_match&current=temperature_2m,apparent_temperature,weather_code&past_days=0&forecast_days=1')
        d = data.json()
        current_weather = d.get('current')
        temp = round(current_weather.get('temperature_2m',0))
        weather_type = WMO_CODES.get(current_weather.get('weather_code'))
        weather = {
            "weather": weather_type,
            "temperature": temp,
        }
        return weather
    except Exception as e:
        print(f"failed to get weather{e}")
        return weather
def weather_prompt() -> str:
    weather = get_weather()
    if weather:
        return f"W Gdańsku aktualnie jest {weather['weather']} i {str(weather['temperature'])} stopni Celsjusza"
    else:
        return "Niestety nie udało się uzyskać danych o pogodzie."
print(weather_prompt())
