import python_weather
import asyncio
'''
Usage 
import -> from weather import get_weather
weather = asyncio.run(get_weather(city)) <- default Gdańsk
returns a dict, kind is mapped to polish
dict contents kind,temperature,feels_like,humidity,pressure,wind_speed,ultraviolet,visibility
get_weather(city) <-default Gdańsk 
returns a sentence
'''
WEATHER_MAP = {
    "Sunny":                  "Słonecznie",
    "Partly Cloudy":          "Częściowe zachmurzenie",
    "Cloudy":                 "Zachmurzenie",
    "Very Cloudy":            "Całkowite zachmurzenie",
    "Fog":                    "Mgła",
    "Light Rain":             "Lekki deszcz",
    "Light Showers":          "Przelotny deszcz",
    "Heavy Rain":             "Ulewny deszcz",
    "Heavy Showers":          "Ulewne opady",
    "Light Sleet":            "Lekka mżawka",
    "Light Sleet Showers":    "Przelotna mżawka",
    "Light Snow":             "Lekki śnieg",
    "Light Snow Showers":     "Przelotny śnieg",
    "Heavy Snow":             "Intensywne opady śniegu",
    "Heavy Snow Showers":     "Ulewne opady śniegu",
    "Thundery Showers":       "Burza z deszczem",
    "Thundery Heavy Rain":    "Gwałtowna burza z deszczem",
    "Thundery Snow Showers":  "Burza ze śniegiem",
}

async def get_weather(city: str = 'Gdansk'):
    async with python_weather.Client() as client:
        try:
            weather = await client.get(city)
            temp = weather.temperature
            feels_like = weather.feels_like
            humidity = weather.humidity
            pressure = weather.pressure
            wind_speed = weather.wind_speed
            ultraviolet = weather.ultraviolet
            visibility = weather.visibility
            kind = WEATHER_MAP.get(weather.kind.name.replace('_',' ').title())
            return {
                "kind" : kind,
                "temperature" : temp,
                "feels_like" : feels_like,
                "humidity" : humidity,
                "pressure" : pressure,
                "wind_speed" : wind_speed,
                "ultraviolet" : ultraviolet,
                "visibility" : visibility,
            }
        except Exception as e:
            print(f"Failed to get weather {e}")
            return {}
'''
Used to inject weather data in polish into LLM prompt defaults to Gdansk
returns a sentence with weather kind mapped into polish.
Can be expanded to give the model more data pressure etc.
'''
def weather_prompt(city: str = 'Gdansk'):
    weather = asyncio.run(get_weather(city))
    return f"W Gdańsku aktualnie jest {weather['kind']} i temperatura {weather['temperature']}C, temperatura odczuwalna {weather['feels_like']}C"
if __name__ == '__main__':
    pass
    # print("Getting weather...")
    # weather = asyncio.run(get_weather('Gdansk'))
    # print(weather['kind'])