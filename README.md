# Senior-Digest
News Made Easy: Keeping Seniors Connected and Informed

## File Information
1. `app.py` : streamlit webapp with summary and query endpoints
2. `fill-1month-news.py` : give the news bot basic knowledge by populating pinecone db with news from past month
3. `update-daily-news.py` : add top 10 daily news to pinecone db and generate summary report to GSheets
4. `view-dashboard.py`: view trulens eval dashboard, that is connected to the online webapp by a remote SQL database

## Project Description:
Senior-Digest is an innovative web application designed to simplify news consumption, making it more accessible and personalized. Imagine a dashboard where, in one tab labeled "Summary," the app neatly displays top 10 news headlines with generated summaries. Users can then switch to the "Query" tab to ask questions that dive into particular topics, either by typing or speaking. Here, the app processes the queries using smart language models and quickly fetches relevant answers from its database. Before showing these answers to the user, it checks them for accuracy and relevance. 

Senior-Digest uses several key technologies to work smoothly and efficiently. Streamlit makes the app's interface easy to use and interactive. Google Cloud offers VertexAI models needed by the application, like LLM, embeddings, speech-to-text and text-to-speech APIs. gspread connects the app to Google Sheets for retrieval of daily news summary. Pinecone is used for storing news embeddings over a period of time. Langchain helps to integrate summarization and RAG components. Finally, Trulens evaluates the query and responses for relevance and groundedness. 

Senior-Digest's features are especially beneficial for the elderly. Its simple access to the top 10 daily news stories and AI-generated summaries reduces the need for lengthy reading, ideal for those who might struggle with small text. Additionally, its voice-based query system and audio outputs make news consumption easier and more accessible for elderly users, accommodating potential visual challenges.
