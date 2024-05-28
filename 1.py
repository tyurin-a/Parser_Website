import requests

url = "https://витрина.фрт.рф/opendata?gid=2208161&page=1&pageSize=32"
response = requests.get(url)
print(response.text)