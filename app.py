import streamlit as st
from st_audiorec import st_audiorec

import wave
from io import BytesIO

import gspread
from google.cloud import aiplatform, speech, texttospeech
from google.oauth2.service_account import Credentials

import pinecone
import langchain
from langchain.vectorstores import Pinecone
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import numpy as np
from trulens_eval import Select, feedback, Feedback, Tru, TruChain, LiteLLM


langchain.debug = True
st.set_page_config('News-Digest', '📰')
st.title('📰 News-Digest')


def setup_feedbacks():
    tru = Tru(database_url=st.secrets['TRULENS_DB_URL'])
    tru.reset_database()
    llm_provider = LiteLLM(model_engine="chat-bison")
    
    # Question/answer relevance between overall question and answer.
    qa_relevance = Feedback(llm_provider.relevance,
                            name="Answer Relevance").on_input_output()

    # Context relevance between question and each context chunk.
    qs_relevance = Feedback(llm_provider.qs_relevance,
                            name="Context Relevance").on_input().on(
                                Select.Record.app.combine_documents_chain._call.
                                args.inputs.input_documents[:].page_content
                            ).aggregate(np.mean)

    # Define groundedness
    grounded = feedback.Groundedness(groundedness_provider=llm_provider)
    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.
            input_documents[:].page_content.collect()
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
    )

    feedback_functions = [qa_relevance, qs_relevance, groundedness]
    return feedback_functions

@st.cache_resource
def load_resources():
    scopes = [
        'https://www.googleapis.com/auth/cloud-platform',
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    service_acc = dict(st.secrets['service_account'])
    my_credentials = Credentials.from_service_account_info(service_acc, scopes=scopes)
    aiplatform.init(credentials=my_credentials, project=service_acc['project_id'])

    llm = VertexAI()
    embeddings = VertexAIEmbeddings()
    gsheet = gspread.authorize(credentials=my_credentials)
    stt = speech.SpeechClient(credentials=my_credentials)
    tts = texttospeech.TextToSpeechClient(credentials=my_credentials)

    template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, 
    don't try to make up an answer. Keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)

    feedbacks = setup_feedbacks()

    return gsheet, llm, embeddings, prompt, stt, tts, feedbacks

def speak(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3)

    response = tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    return response.audio_content

def get_audio_configs(audio_bytes):
    audio_file = wave.open(BytesIO(audio_bytes))
    frame_rate = audio_file.getframerate()
    channels = audio_file.getnchannels()
    return frame_rate, channels

def listen(voice):
    audio = speech.RecognitionAudio(content=voice)
    frame_rate, channels = get_audio_configs(voice)

    config = speech.RecognitionConfig(
        encoding = 'LINEAR16',
        language_code = 'en-US',
        sample_rate_hertz = frame_rate,
        audio_channel_count = channels,
    )

    response = stt.recognize(config=config, audio=audio)
    query = response.results[0].alternatives[0].transcript
    return query


gsheet, llm, embeddings, prompt, stt, tts, feedbacks = load_resources()

if 'news_data' not in st.session_state:
    sheet = gsheet.open('Daily News Summary').sheet1
    st.session_state.news_data = sheet.get('A2:D11')

if 'vector_store' not in st.session_state:
    pinecone.init(
        api_key=st.secrets['PINECONE_API_KEY'],
        environment='gcp-starter',
    )
    st.session_state.vector_store = Pinecone.from_existing_index('news-data', embeddings)


tabs = st.tabs(['Summary', 'Query'])
with tabs[0]:
    for i, news in enumerate(st.session_state.news_data):
        with st.expander(news[0]):
            st.caption(f'Published by {news[1]} on {news[2]}')
            st.write(news[3])
            if st.button('play audio', key=i):
                with st.spinner('processing audio..'):
                    voice = speak(news[3])
                    st.audio(voice)

with tabs[1]:
    media = st.radio('Choose input method:', ['speak', 'type'])
    if media == 'type':
        query = st.text_input('Write your query!')
    else:
        audio = st_audiorec()
        if audio:
            query = listen(audio)
            st.write(f'Query: {query}')
            st.divider()
        else:
            query = ''

    if query:
        qa_chain = RetrievalQA.from_chain_type(   
            llm=llm,   
            chain_type="stuff", 
            retriever=st.session_state.vector_store.as_retriever(),   
            chain_type_kwargs={"prompt": prompt}
        )

        chain_recorder = TruChain(
            qa_chain,
            app_id="News-Digest",
            feedbacks=feedbacks
        )

        with chain_recorder as recording:
            response = qa_chain.run(query)
        
        st.write(f'Response: {response}')
        voice = speak(response)
        st.audio(voice)
