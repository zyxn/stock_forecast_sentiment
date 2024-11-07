from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
import pandas as pd
from tqdm import tqdm
import asyncio

# os.environ["OPENAI_API_KEY"] = "api"





# Define the desired data structure.
class NewsAnalysis(BaseModel):
    sentimen_berita: str = Field(description="Sentimen berita yang berupa positif, negatif, atau netral")
    skor_sentimen: float = Field(description="Skor sentimen antara -1 dan 1")
    ner: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "ORG": [],
            "PER": [],
            "LOC": [],
            "MONEY": [],
            "SEC": [],
            "DATE/TIME": [],
            "EVENT": [],
            "ACTION": [],
            "INSTRUMENT": [],
            "TREND": [],
            "PERCENT": [],
            "ECON_IND": [],
            "LEGAL": []
        },
        description=(
            "Ekstraksi entitas terkait keuangan, dengan kategori entitas: ORG (Organisasi), PER (Tokoh), LOC (Lokasi), "
            "MONEY (Jumlah uang), SEC (Saham & Sekuritas), DATE/TIME (Tanggal/Waktu), EVENT (Acara ekonomi), ACTION (Aksi finansial), "
            "INSTRUMENT (Instrumen keuangan), TREND (Tren pasar), PERCENT (Persentase), ECON_IND (Indikator ekonomi), LEGAL (Terminologi hukum)"
        )
    )
    keypoint_berita: str = Field(description="Ringkasan singkat dari berita dalam 2-3 kalimat")
    top_3_words: List[str] = Field(description="Tiga kata paling sering muncul dalam berita yang relevan")


model = ChatOpenAI(model="gpt-4o-mini",temperature=0).bind_tools([NewsAnalysis])

# Define the query intended for news extraction and analysis.
# news_query = "IHSG Bakal Hadapi Tekanan Ambil Untung  IHSG Bakal Hadapi Tekanan Ambil Untung  Penulis JAKARTA, KOMPAS.com -  Tekanan ambil untung bakal mewarnai pergerakan Indeks Harga Saham Gabungan, Selasa (21/5/2013), pascamencetak rekor tertinggi baru. Pergerakan bursa Asia pun akan memengaruhi gerak indeks hari ini. Bursa saham Wall Street ditutup melemah semalam waktu Indonesia, memberikan sentimen negatif bagi pasar. Ini setelah kekhawatiran dipercepatnya pengurangan dan bahkan kemungkinan penghentian program stimulus di Amerika Serikat. Indeks Dow Jones Industrial Average melemah 0,12 persen ke level 15.335; Indeks S&P500 turun tipis 0,07 persen ke level 1.666 dan Indeks Komposit Nasdaq melemah tipis 0,07 persen ke level 3.496. Pada perdagangan kemarin, IHSG ditutup naik 69,29 poin (1,35 persen) ke level 5.214,98 dengan jumlah transaksi sebanyak 11,9 juta lot atau setara dengan Rp 7,5 triliun. Investor asing tercatat melakukan pembelian bersih di pasar reguler sebesar Rp 571 miliar dengan saham yang paling banyak dibeli antara lain BMRI,Â ASII, MAIN, INDF dan BBRI. Mata uang rupiah terdepresiasi ke level Rp 9.758 per dollar AS. Secara teknikal, menurut riset eTrading Securities, kenaikan IHSG menghasilkan sinyal bullish pada indikator MACD. Hal ini terlihat dari histogram yang menciptakan new high mengindikasikan penguatan akan berlanjut pada hari ini. Dengan dukungan di level 5.075 dan resisten di level 5.300. Adapun saham-saham yang dapat diperhatikan adalah BMRI, GGRM dan MAIN.                Penulis JAKARTA, KOMPAS.com -  Tekanan ambil untung bakal mewarnai pergerakan Indeks Harga Saham Gabungan, Selasa (21/5/2013), pascamencetak rekor tertinggi baru. Pergerakan bursa Asia pun akan memengaruhi gerak indeks hari ini. Bursa saham Wall Street ditutup melemah semalam waktu Indonesia, memberikan sentimen negatif bagi pasar. Ini setelah kekhawatiran dipercepatnya pengurangan dan bahkan kemungkinan penghentian program stimulus di Amerika Serikat. Indeks Dow Jones Industrial Average melemah 0,12 persen ke level 15.335; Indeks S&P500 turun tipis 0,07 persen ke level 1.666 dan Indeks Komposit Nasdaq melemah tipis 0,07 persen ke level 3.496. Pada perdagangan kemarin, IHSG ditutup naik 69,29 poin (1,35 persen) ke level 5.214,98 dengan jumlah transaksi sebanyak 11,9 juta lot atau setara dengan Rp 7,5 triliun. Investor asing tercatat melakukan pembelian bersih di pasar reguler sebesar Rp 571 miliar dengan saham yang paling banyak dibeli antara lain BMRI,Â ASII, MAIN, INDF dan BBRI. Mata uang rupiah terdepresiasi ke level Rp 9.758 per dollar AS. Secara teknikal, menurut riset eTrading Securities, kenaikan IHSG menghasilkan sinyal bullish pada indikator MACD. Hal ini terlihat dari histogram yang menciptakan new high mengindikasikan penguatan akan berlanjut pada hari ini. Dengan dukungan di level 5.075 dan resisten di level 5.300. Adapun saham-saham yang dapat diperhatikan adalah BMRI, GGRM dan MAIN.                Penulis JAKARTA, KOMPAS.com -  Tekanan ambil untung bakal mewarnai pergerakan Indeks Harga Saham Gabungan, Selasa (21/5/2013), pascamencetak rekor tertinggi baru. Pergerakan bursa Asia pun akan memengaruhi gerak indeks hari ini. Bursa saham Wall Street ditutup melemah semalam waktu Indonesia, memberikan sentimen negatif bagi pasar. Ini setelah kekhawatiran dipercepatnya pengurangan dan bahkan kemungkinan penghentian program stimulus di Amerika Serikat. Indeks Dow Jones Industrial Average melemah 0,12 persen ke level 15.335; Indeks S&P500 turun tipis 0,07 persen ke level 1.666 dan Indeks Komposit Nasdaq melemah tipis 0,07 persen ke level 3.496. Pada perdagangan kemarin, IHSG ditutup naik 69,29 poin (1,35 persen) ke level 5.214,98 dengan jumlah transaksi sebanyak 11,9 juta lot atau setara dengan Rp 7,5 triliun. Investor asing tercatat melakukan pembelian bersih di pasar reguler sebesar Rp 571 miliar dengan saham yang paling banyak dibeli antara lain BMRI,Â ASII, MAIN, INDF dan BBRI. Mata uang rupiah terdepresiasi ke level Rp 9.758 per dollar AS. Secara teknikal, menurut riset eTrading Securities, kenaikan IHSG menghasilkan sinyal bullish pada indikator MACD. Hal ini terlihat dari histogram yang menciptakan new high mengindikasikan penguatan akan berlanjut pada hari ini. Dengan dukungan di level 5.075 dan resisten di level 5.300. Adapun saham-saham yang dapat diperhatikan adalah BMRI, GGRM dan MAIN.                JAKARTA, KOMPAS.com -  Tekanan ambil untung bakal mewarnai pergerakan Indeks Harga Saham Gabungan, Selasa (21/5/2013), pascamencetak rekor tertinggi baru. Pergerakan bursa Asia pun akan memengaruhi gerak indeks hari ini. Bursa saham Wall Street ditutup melemah semalam waktu Indonesia, memberikan sentimen negatif bagi pasar. Ini setelah kekhawatiran dipercepatnya pengurangan dan bahkan kemungkinan penghentian program stimulus di Amerika Serikat. Indeks Dow Jones Industrial Average melemah 0,12 persen ke level 15.335; Indeks S&P500 turun tipis 0,07 persen ke level 1.666 dan Indeks Komposit Nasdaq melemah tipis 0,07 persen ke level 3.496. Pada perdagangan kemarin, IHSG ditutup naik 69,29 poin (1,35 persen) ke level 5.214,98 dengan jumlah transaksi sebanyak 11,9 juta lot atau setara dengan Rp 7,5 triliun. Investor asing tercatat melakukan pembelian bersih di pasar reguler sebesar Rp 571 miliar dengan saham yang paling banyak dibeli antara lain BMRI,Â ASII, MAIN, INDF dan BBRI. Mata uang rupiah terdepresiasi ke level Rp 9.758 per dollar AS. Secara teknikal, menurut riset eTrading Securities, kenaikan IHSG menghasilkan sinyal bullish pada indikator MACD. Hal ini terlihat dari histogram yang menciptakan new high mengindikasikan penguatan akan berlanjut pada hari ini. Dengan dukungan di level 5.075 dan resisten di level 5.300. Adapun saham-saham yang dapat diperhatikan adalah BMRI, GGRM dan MAIN.  Copyright 2008 - 2024 PT. Kompas Cyber Media (Kompas Gramedia Digital Group). All Rights Reserved."

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputKeyToolsParser(key_name="NewsAnalysis", first_tool_only=True)

prompt = ChatPromptTemplate.from_messages(
    [("system", "you are proffesor in economics and stocks"), ("user", "{input}")]
)

chain = prompt | model | parser

# Example invocation (replace 'news_query' with actual news content to analyze)
# print(chain.invoke({"input": news_query})[0]['args'])

data = pd.read_csv('result_fix_sample10.csv',sep=";")
async def process_all_data(batch_size=5):
    results = []
    num_articles = len(data)
    
    # Split data into batches
    batches = [data['isi'][i:i + batch_size].tolist() for i in range(0, num_articles, batch_size)]
    
    for batch in tqdm(batches, desc="Analyzing articles", unit="batch"):
        # Use chain.asyncbatch to process the batch
        batch_results = await chain.abatch([{"input": content} for content in batch])
        results.extend(batch_results)
        await asyncio.sleep(1)
    # Create a new column in DataFrame for extracted results
    data["Extract"] = results
    # Save to a new CSV
    data.to_csv("result_all_Extract_fix.csv", index=False)

# Run the async processing
asyncio.run(process_all_data(batch_size=5))