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
    48: "Mgła osadzająca szadź",
    51: "Lekka mżawka",
    53: "Umiarkowana mżawka",
    55: "Gęsta mżawka",
    56: "Lekka marznąca mżawka",
    57: "Gęsta marznąca mżawka",
    61: "Lekki deszcz",
    63: "Umiarkowany deszcz",
    65: "Ulewny deszcz",
    66: "Lekki marznący deszcz",
    67: "Ulewny marznący deszcz",
    71: "Lekkie opady śniegu",
    73: "Umiarkowane opady śniegu",
    75: "Intensywne opady śniegu",
    77: "Śnieg ziarnisty",
    80: "Lekki przelotny deszcz",
    81: "Umiarkowany przelotny deszcz",
    82: "Gwałtowny przelotny deszcz",
    85: "Lekkie przelotne opady śniegu",
    86: "Intensywne przelotne opady śniegu",
    95: "Burza",
    96: "Burza z lekkim gradem",
    99: "Burza z silnym gradem",
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
        return f"W Gdańsku aktualnie jest {weather['weather']} i {str(weather['temperature'])} stopni Celsjusza."
    else:
        return "Niestety nie udało się uzyskać danych o pogodzie."
