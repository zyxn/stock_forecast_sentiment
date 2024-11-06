import json
from selectolax.parser import HTMLParser
from dataclasses import dataclass, asdict
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import httpx
import random
from httpx_socks import SyncProxyTransport
import logging
import os

logging.basicConfig(
    format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO  # Set ke INFO untuk mengurangi output log
)


@dataclass
class News:
    tanggal: str
    judul: str
    link: str
    isi: str
    
    
    

def get_html(url, client, retries=3, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    
    for attempt in range(retries):
        try:
            resp = client.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 403:
                logging.warning(f"Access denied (403) on {url}. Stopping scraper.")
                return None, 403
            return HTMLParser(resp.text), resp.status_code
        except httpx.ReadTimeout:
            logging.warning(f"Timeout on {url}, attempt {attempt + 1}/{retries}")
            time.sleep(2)
        except httpx.RequestError as e:
            logging.error(f"Request error on {url}: {e}")
            break
    
    return None, None

def collect_links(pages, client):
    all_links = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(get_html, f"https://www.kompas.com/tag/ihsg?sort=asc&page={page}", client)
            for page in range(1, pages + 1)
        ]
        
        for index, future in enumerate(tqdm(futures, desc="Getting All links Page"), start=1):
            html = future.result()
            if html:
                news = html.css("div.article__list")
                for x in news:
                    link = x.css_first("a").attributes.get("href", "")
                    if link:
                        link = link.replace("bisniskeuangan", "money").replace("http", "https")
                        all_links.append(link)
                logging.info(f"Processed page {index} with {len(news)} articles.")
            else:
                logging.warning(f"Failed to retrieve content on page {index}.")
            
    return all_links

def get_news_data(link, client):
    short_link = link[:20] + '...'  # Singkatkan link untuk logging
    logging.info(f"Processing link: {short_link}")
    resp = client.get(link)
    page = HTMLParser(resp.text)
    tags_to_strip = ['style', 'script', 'xmp', 'iframe', 'noembed', 'noframes']
    page.strip_tags(tags_to_strip)
    try:
        tanggal = page.css_first('div.read__time').text() if page.css_first('div.read__time') else "N/A"
        judul = page.css_first('h1.read__title').text() if page.css_first('h1.read__title') else "N/A"
        isi = "".join(i.text() for i in page.css('div.clearfix p')) or "N/A"
        
        new_data = News(
            tanggal=tanggal,
            judul=judul,
            link=link,
            isi=isi
        )
        return new_data
    except Exception as e:
        logging.error(f"Error processing link {short_link}: {e}")
        return News(tanggal="error", judul="error", link=link, isi="error")

def write_to_csv(file_path, news_list):
    with open(file_path, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=['tanggal', 'judul', 'link', 'isi'])
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerows(asdict(news) for news in news_list)

def process_link(link, client):
    news = get_news_data(link, client)
    return news

















def main():
    start_time = time.time()
    results = []
    
    # Load proxies from a JSON file
    if not os.path.exists("./result_mahal.csv"):
    # Create an httpx Client with SOCKS5 proxy
        with httpx.Client(timeout=10.0) as client:
            links = collect_links(10, client)
            
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(process_link, link, client) for link in links]
                
                for index, future in enumerate(tqdm(futures, total=len(links), desc="Pages")):
                    results.append(future.result())
                    
            write_to_csv("./result_mahal.csv", results)
    else:
        pass
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Execution Time: {execution_time} seconds")

main()
