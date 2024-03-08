import requests
from bs4 import BeautifulSoup
import pandas as pd
import httpx
from tqdm import tqdm
import asyncio

data = {'Tanggal': [], 'Judul': [], 'Link': [], 'Isi': []}
def main():
    for i in tqdm(range(1, 1001), desc="Scraping Pages"):
        home = f"https://www.cnbcindonesia.com/tag/ihsg/{i}?kanal=&tipe="
        async with httpx.AsyncClient() as client:
            response = await client.get(home)

        soup = BeautifulSoup(response.text, 'html.parser')

        article_elements = soup.find_all('article')

        for article in tqdm(article_elements, desc=f"Processing Page {i}"):
            link = article.find("a")["href"]
            judul = article.find("h2").get_text()
            data['Link'].append(link)
            data['Judul'].append(judul)
            link_full = link + "?page=all"

            # async with httpx.AsyncClient() as client:
            #     response2 = await client.get(link_full)

            # soup2 = BeautifulSoup(response2.text, 'html.parser')
            # detail_text = soup2.find('div', class_="detail_text")
            # isi = "".join([i.get_text() for i in detail_text.find_all('p')])
            # tanggal = soup2.find('div', class_="date").get_text()
            # data['Isi'].append(isi)
            # data['Tanggal'].append(tanggal)

    df = pd.DataFrame(data)
    return df

asyncio.run(main())