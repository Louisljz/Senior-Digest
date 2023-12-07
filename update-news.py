import os
import re
import json
from tqdm import tqdm
from urllib import request
from datetime import datetime
from dotenv import load_dotenv

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

import gspread
import pinecone
from langchain.vectorstores import Pinecone

import warnings
warnings.filterwarnings('ignore')
load_dotenv()


def generate_summary(article):
    chain = load_summarize_chain(llm,
                                chain_type="stuff",
                                prompt=prompt)
    summary = chain.run(article)
    return summary

# Initiate Langchain components
prompt_template = """
Generate summary for the following text, using the following steps:
1. summary consists of maximum 100 words
2. If the text cannot be found or error, return: "Content empty"
3. Use only materials from the text supplied
4. Create summary in English

"{text}"
SUMMARY: """
prompt = PromptTemplate.from_template(prompt_template)

llm = VertexAI()
embeddings = VertexAIEmbeddings()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment='gcp-starter',
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)

# Authenticate google-spreadsheets
service_acc = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
gc = gspread.service_account(filename=service_acc)
sheet = gc.open('Daily News Summary').sheet1
sheet.batch_clear(['A2:D11'])

# Fetch articles from GNEWS
api_key = os.getenv('GNEWS_API_KEY')
today = datetime.today().strftime(r"%Y-%m-%d")
url = f"https://gnews.io/api/v4/top-headlines?&lang=en&max=10&from={today}&apikey={api_key}"
with request.urlopen(url) as response:
    data = json.loads(response.read().decode("utf-8"))
    articles = data['articles']

# Updating GSheet and Zilliz DB
news_data = []
documents = []
for news in tqdm(articles, desc='Processing Articles'):
    loader = WebBaseLoader(news['url'])
    doc = loader.load()
    doc[0].page_content = re.sub(r'\s+', ' ', doc[0].page_content)
    
    summary = generate_summary(doc)

    news_data.append([
        news['title'],
        news['source']['name'],
        news['url'],
        summary
    ])

    documents.extend(doc)

# update gsheet
sheet.update('A2', news_data)

# update vector db
print('Pushing news documents to vectorDB')
doc_splits = text_splitter.split_documents(documents)
docsearch = Pinecone.from_documents(doc_splits, embeddings, index_name='news-data')

print("GSheet and Pinecone DB updated with news today!")
