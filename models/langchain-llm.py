from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

def analyze_article(article_text):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": "Analyze the sentiment of a stock news article, then extract essential content such as keywords, key points, top 3 words, the sentiment score, Named Entity Recognition (NER), and polarity if applicable.\n\nFollow these steps to ensure thorough analysis:\n\n1. **Sentiment Analysis**: Determine if the overall sentiment of the news article is positive, negative, or neutral. Additionally, provide a float value between -1 and 1 as the sentiment score to indicate the strength of the sentiment. Use the context of the stock market to provide an accurate sentiment rating.\n\n2. **Content Extraction**:\n    - Extract keywords that capture the core idea of the article (3-5 keywords).\n    - Identify and summarize key points that convey the most important aspects of the news.\n  \n3. **Top 3 Words**:\n    - Identify the top 3 words used in the article that are most relevant or repeated, excluding common stopwords.\n\n4. **Named Entity Recognition (NER)**: Identify and extract significant entities from the article, such as company names, person names, dates, monetary values, etc.\n\n5. **Polarity Extraction (if applicable)**:\n    - Extract the polarity of the news article, specifically indicating if the content includes polarizing or controversial information. Use labels such as “polarizing” or “neutral”.\n\n# Output Format\n\nProduce the output as a well-structured JSON with the following fields:\n\n```json\n{\n  \"sentiment\": {\n    \"label\": \"[positive/negative/neutral]\",\n    \"score\": float_value\n  },\n  \"keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"],\n  \"key_points\": [\n    \"Point 1 summarizing an important aspect of the news\",\n    \"Point 2 providing additional detail or context\",\n    \"Point 3 final key element\"\n  ],\n  \"top_3_words\": [\"word1\", \"word2\", \"word3\"],\n  \"named_entities\": [\n    {\n      \"entity\": \"entity_name\",\n      \"type\": \"entity_type\"\n    }\n  ],\n  \"polarity\": \"[polarizing/neutral]\"\n}\n```\n\n# Notes\n\n- Make sure to distinguish important words by their relevance to the stock market, ignoring common words like \"the\", \"is\", or irrelevant filler words.\n- Extract concise and meaningful keywords that represent the primary subject matter of the news article.\n- In the NER section, include types like \"Company\", \"Person\", \"Monetary Value\", \"Date\" among others.\n- Polarity should only be included if the article suggests a polarizing or controversial situation.",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": article_text
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_object"
    }
    )

    return response.choices[0].message.content

data = pd.read_csv('result_fix_sample10.csv',sep=";")

results = []
for content in tqdm(data['isi'], desc="Analyzing articles", unit="article"):
    # print(content)
    result = analyze_article(content)
    results.append(result)
# Create DataFrame with results

data["extract"] = result
data.to_csv("result_sample_withExtarct.csv", index=False)