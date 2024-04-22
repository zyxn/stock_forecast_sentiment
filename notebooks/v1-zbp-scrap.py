import httpx
from selectolax.parser import HTMLParser
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor

@dataclass
class News:
    tanggal: str
    judul: str
    link: str
    isi: str

def get_html_link(page, client):
    url = f"https://www.cnbcindonesia.com/tag/ihsg/{page}?kanal=&tipe="
    resp = client.get(url)
    return HTMLParser(resp.text)

def collect_link(page, client):
    all_links = []
    for page_num in tqdm(range(1, page+1), desc="Getting All links Page"):
        html = get_html_link(page_num, client)
        news = html.css("article")
        for x in news:
            hasil = x.css_first("a").attributes["href"]
            all_links.append(hasil + "?page=all")
    return all_links

def get_news_data(link, client):
    resp = client.get(link)
    page = HTMLParser(resp.text)
    tags = ['style', 'script', 'xmp', 'iframe', 'noembed', 'noframes']
    page.strip_tags(tags)
    
    meta_isi = page.css_first("div.detail_text")
    try: 
        new_data = News(
            tanggal=page.css_first("div.date").text(),
            judul=page.css_first("title").text(),
            link=link,
            isi="".join([i.text() for i in meta_isi.css('p')])
        )
        return new_data
    except:
        new_data = News(
            tanggal="error",
            judul="error",
            link=link,
            isi="error"
        )
        print(link)
        return new_data

def to_csv(res):
    with open("result.csv", "a", encoding='utf-8') as f:
        writer = csv.DictWriter(f, delimiter=';', quotechar='|', fieldnames=['tanggal', 'judul', 'link', 'isi'])
        writer.writeheader()
        writer.writerows(res)

def process_link(link):
    global client
    news = get_news_data(link, client)
    return asdict(news)

def main():
    global client
    results = []
    with httpx.Client() as client:
        links = collect_link(1000, client)
        with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust max_workers as needed
            futures = [executor.submit(process_link, link) for link in links]
            for future in tqdm(futures, total=len(links), desc="Pages"):
                results.append(future.result())
    to_csv(results)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Time Exec:", execution_time, "seconds")

#Using Multi Proccesing dont forget adjust workers
#TODO : to improve get the all links use multiprocces too
# ~ 5 mins to scrap 10k data