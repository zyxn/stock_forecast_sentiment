import httpx
from selectolax.parser import HTMLParser
from dataclasses import dataclass, asdict
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

@dataclass
class News:
    tanggal: str
    judul: str
    link: str
    isi: str

def get_html(url, client):
    resp = client.get(url)
    return HTMLParser(resp.text)

def collect_links(pages, client):
    all_links = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(get_html, f"https://www.cnbcindonesia.com/tag/ihsg/{page}?kanal=&tipe=", client) for page in range(1, pages+1)]
        for future in tqdm(futures, desc="Getting All links Page"):
            html = future.result()
            news = html.css("article")
            for x in news:
                link = x.css_first("a").attributes["href"]
                all_links.append(link + "?page=all")
    return all_links

def get_news_data(link, client):
    resp = client.get(link)
    page = HTMLParser(resp.text)
    tags_to_strip = ['style', 'script', 'xmp', 'iframe', 'noembed', 'noframes']
    page.strip_tags(tags_to_strip)
    try:
        new_data = News(
            tanggal=page.css_first("div.date").text(),
            judul=page.css_first("title").text(),
            link=link,
            isi="".join(i.text() for i in page.css('div.detail_text p'))
        )
        return new_data
    except Exception as e:
        print(f"Error processing link {link}: {e}")
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
    with httpx.Client() as client:
        links = collect_links(1000, client)  # Edit to maximum number of pages
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_link, link, client) for link in links]
            for future in tqdm(futures, total=len(links), desc="Pages"):
                results.append(future.result())
        write_to_csv("./result.csv", results)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Time Exec:", execution_time, "seconds")

if __name__ == '__main__':
    main()

#Refactored
#Time improvement from ~5mins to ~2mins for 10k news data