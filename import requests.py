import requests
from bs4 import BeautifulSoup
import os
import tempfile
import zipfile
from time import sleep

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def get_url():
    for count in range(1, 15):
        
        url = f"https://витрина.фрт.рф/opendata?gid=2208161&page={count}&pageSize=32"
        
        response = requests.get(url, headers=headers)

        #print(response.encoding)
        # prints out ISO-8859-1
        #response.encoding = 'utf-8'  # override encoding manually
        sleep(1)
        soup = BeautifulSoup(response.text, features="lxml")
        #print(soup.prettify())

        data_filter = soup.find_all("div", class_="lh-27 f-28 fw-500 mt-48")
        for i in data_filter:
            data_name = i.text
            if data_name.startswith("Реестр домов по "):
                data = i.find_next_sibling("div", class_ = "row mt-24")
                print(data)
                for j in data:
                    res = soup.find("a", class_="green-link-only-hover f-12 fw-600 ml-2 text-uppercase", string = "Экспорт")
                    print(res)
        #for item in data:
        #    zip_url = "https://витрина.фрт.рф" + item.find("a").get("href")
            #download_zip(zip_url)
        #    print(zip_url)

def download_zip(url):
    response = requests.get(url, headers=headers)
    file = tempfile.TemporaryFile()
    file.write(response.content)
    fzip = zipfile.ZipFile(file)
    fzip.extractall(r"C:\Users\a.tyurin\Desktop\Results")
    file.close()
    fzip.close() 

get_url()
print("Done!")