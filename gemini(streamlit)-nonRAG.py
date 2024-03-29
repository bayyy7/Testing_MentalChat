import os
from dotenv import load_dotenv
import streamlit as st
from dataclasses import dataclass
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

GEMINI_API_KEY=st.secrets['GEMINI_API_KEY']

st.set_page_config(page_title="💬 MentalChat")

with st.sidebar:
    st.title('💬 :rainbow[MentalChat]')
    st.markdown('---')
    st.markdown('# About')
    st.markdown(
        'MentalChat adalah AI assistant yang dirancang untuk menjadi pendamping digital bagi mereka yang menghadapi tantangan kesehatan mental. Dengan menggunakan teknologi kecerdasan buatan canggih, MentalChat berfungsi sebagai psikiater dan terapis virtual, memberikan dukungan emosional, saran praktis, dan teknik relaksasi yang dapat diakses kapan saja, di mana saja')
    st.markdown('---')
    st.markdown('# Powered by')
    st.markdown(':grey[Google Gemini Pro]')
    st.markdown('---')
    st.markdown('# Create by')
    st.markdown(':grey[Rizky Indabayu]')
    

@dataclass
class Message:
    actor: str
    payload: str


@st.cache_resource
def get_llm() -> GoogleGenerativeAI:
    
    return GoogleGenerativeAI(model="gemini-pro", 
                        google_api_key=GEMINI_API_KEY,
                        temperature=0.99,
                        top_p=0.6,
                        top_k=1,
                        repeat_penalty=1.1,
                        max_output_tokens=1024)


def get_llm_chain():
    template = """Mulai sekarang, kamu adalah AI asisten bernama 'MentalChat'. Bertindaklah sebagai seorang Psikolog profesional yang dapat 
        melakukan diagnosis masalah kesehatan mental dengan cara mengumpulkan informasi sebanyak-banyak tentang permasalahan pasien. 
        Kunci diagnosis adalah memperoleh gambaran sejelas mungkin tentang gejala yang dialami klien atau pasien. 
        Berdasarkan informasi tersebut, Kamu bisa melakukan diagnosa secara akurat dan berikan solusi pengobatan yang tepat untuk pasien.
        Kemudian, untuk pertanyaan diluar konteks kesehatan mental jawablah dengan 'Maaf, masukan Anda diluar konteks kesehatan mental'.
    
    Previous conversation:
    {chat_history}
    
    Human question:{question}
    Response:"""
    prompt_template = PromptTemplate.from_template(template)
    # Notice that we need to align the `memory_key`
    memory = ConversationBufferMemory(memory_key="chat_history")
    conversation = LLMChain(
        llm=get_llm(),
        prompt=prompt_template,
        verbose=True,
        memory=memory,
    )
    return conversation


USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"


def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Selamat datang di MentalChat. Sebelum kita mulai, apakah ada topik atau isu khusus yang ingin Anda diskusikan atau bahas hari ini? Jangan ragu untuk berbagi perasaan, pemikiran, atau pengalaman yang mungkin ingin Anda eksplorasi bersama.")]
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm_chain()


def get_llm_chain_from_session() -> LLMChain:
    return st.session_state["llm_chain"]


initialize_session_state()

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)

    with st.spinner("Please wait.."):
        llm_chain = get_llm_chain_from_session()
        response: str = llm_chain({"question": prompt})["text"]
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
        st.chat_message(ASSISTANT).write(response)