import pandas as pd
import httpx
import logging
import os
from selectolax.parser import HTMLParser
from urllib.parse import urljoin
import time
from concurrent.futures import ThreadPoolExecutor

# Set up logging configuration
def setup_logging():
    logging.basicConfig(
        format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )

class NewsExtractor:
    def __init__(self, base_url, max_retries=3, timeout=10):
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.cache = {}
        self.total_pages = 0

    def get_html(self, url):
        if url in self.cache:
            return self.cache[url]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        for attempt in range(self.max_retries):
            try:
                resp = self.client.get(url, headers=headers)
                resp.raise_for_status()
                self.cache[url] = HTMLParser(resp.text)
                return self.cache[url]
            except httpx.HTTPStatusError as e:
                logging.warning(f"HTTP error {e.response.status_code} on {url}: {e}")
            except httpx.RequestError as e:
                logging.error(f"Request error on {url}: {e}")
                time.sleep(2)

        self.cache[url] = None
        return None

    def get_news_data(self, link, current_page, total_pages):
        logging.info(f"Processing link:(Page {current_page} of {total_pages}) {link}")
        html = self.get_html(link)
        if html:
            tags_to_strip = ['style', 'script', 'xmp', 'iframe', 'noembed', 'noframes']
            html.strip_tags(tags_to_strip)

            tanggal = html.css_first('div.read__time').text() if html.css_first('div.read__time') else "N/A"
            judul = html.css_first('h1.read__title').text() if html.css_first('h1.read__title') else "N/A"
            isi = " ".join(i.text() for i in html.css('div.clearfix p')) or "N/A"

            return {
                "tanggal": tanggal,
                "judul": judul,
                "link": link,
                "isi": isi
            }
        else:
            logging.info(f"Gagal Processing link:(Page {current_page} of {total_pages}) {link}")
            return {
                "tanggal": "error",
                "judul": "error",
                "link": link,
                "isi": "error"
            }

    def collect_links(self, pages):
        self.total_pages = pages
        all_links = []
        for page in range(1, pages + 1):
            url = urljoin(self.base_url, f"?sort=asc&page={page}")
            html = self.get_html(url)
            if html:
                news = html.css("div.article__list")
                for item in news:
                    link = item.css_first("a").attributes.get("href", "")
                    if link:
                        link = link.replace("bisniskeuangan", "money").replace("http", "https")
                        all_links.append(link)
                logging.info(f"Processed page {page} of {pages} with {len(news)} articles.")
            else:
                logging.warning(f"Failed to retrieve content on page {page}.")
        return all_links

    def close(self):
        self.client.close()

def read_existing_csv(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=';')
        return df.set_index('link').to_dict(orient='index')
    else:
        return {}

def write_to_csv(file_path, news_list):
    # Read existing CSV data into a Pandas DataFrame
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=';')
    else:
        df = pd.DataFrame(columns=['tanggal', 'judul', 'link', 'isi'])

    # Update existing rows or add new rows
    for news in news_list:
        if news['link'] in df['link'].values:
            # Update existing row
            df.loc[df['link'] == news['link'], ['tanggal', 'judul', 'isi']] = [news['tanggal'], news['judul'], news['isi']]
        else:
            # Add new row
            df = pd.concat([df, pd.DataFrame([news])], ignore_index=True)

    # Write the updated DataFrame to the CSV file
    df.to_csv(file_path, sep=';', index=False)

def main(pages=10):
    setup_logging()
    start_time = time.time()
    extractor = NewsExtractor("https://www.kompas.com/tag/ihsg", max_retries=3, timeout=10)

    if not os.path.exists("./result_fix.csv"):
        links = extractor.collect_links(pages)
        with ThreadPoolExecutor(max_workers=16) as executor:
            news_list = list(executor.map(lambda link: extractor.get_news_data(link, links.index(link) + 1, pages), links))
        write_to_csv("./result_fix.csv", news_list)
    else:
        logging.info("CSV file already exists, checking for updates.")
        existing_data = read_existing_csv("./result_fix.csv")
        links_to_update = [link for link in existing_data.keys() if existing_data[link]['tanggal'] == 'error']
        with ThreadPoolExecutor(max_workers=16) as executor:
            news_list = list(executor.map(lambda link: extractor.get_news_data(link, links_to_update.index(link) + 1, len(links_to_update)), links_to_update))
        write_to_csv("./result_fix.csv", news_list)

    extractor.close()
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Execution Time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main(370)
