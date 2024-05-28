import requests
from bs4 import BeautifulSoup
import os

url = "https://витрина.фрт.рф/opendata?gid=2208161&page=9&pageSize=32"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)

#print(response.encoding)
# prints out ISO-8859-1
response.encoding = 'utf-8'  # override encoding manually

soup = BeautifulSoup(response.text, features="lxml")
#print(soup.prettify())

# Check the HTML structure for the correct class and tag
data = soup.find_all("a", class_="green-link-only-hover f-12 fw-600 ml-2 text-uppercase")
for item in data:
    print(item['href'])
