import vector_db
import streamlit as st
from dataclasses import dataclass
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

GEMINI_API_KEY=st.secrets['GEMINI_API_KEY']

vector_db = vector_db.main()

with st.sidebar:
    st.title('💬 :rainbow[MentalChat]')
    st.markdown('---')
    st.markdown('# About')
    st.markdown(
        'MentalChat adalah AI assistant yang dirancang untuk menjadi pendamping digital bagi mereka yang menghadapi tantangan kesehatan mental. Dengan menggunakan teknologi kecerdasan buatan canggih, MentalChat berfungsi sebagai psikiater dan terapis virtual, memberikan dukungan emosional, saran praktis, dan teknik relaksasi yang dapat diakses kapan saja, di mana saja')
    st.markdown('---')
    st.markdown('# Powered by')
    st.markdown(':grey[Google PaLM 2]')
    st.markdown('---')
    st.markdown('# Create by')
    st.markdown(':grey[Rizky Indabayu]')
    

@dataclass
class Message:
    actor: str
    payload: str

@st.cache_resource
def get_llm() -> GoogleGenerativeAI:
    llm = GoogleGenerativeAI(model="gemini-pro", 
                        google_api_key=GEMINI_API_KEY,
                        temperature=0.7,
                        top_p=0.6,
                        top_k=1,
                        repeat_penalty=1.1,
                        max_output_tokens=1024)
    return llm


def prompt():
    template = """Gunakan konteks berikut sebagai referensi untuk menjawab pertanyaan. 
        {context}
        
        Mulai sekarang, kamu adalah AI asisten bernama 'MentalChat'. Bertindaklah sebagai seorang Psikolog profesional yang dapat 
        melakukan diagnosis masalah kesehatan mental dengan cara mengidentifikasi masalah pasien secara bertahap. Berdasarkan identifikasi tersebut,
        berikan jawaban dan saran yang akan membuat pasien merasa lebih baik.
        Question: {question}
        History: {history}
        Answer:"""
        
    prompts = PromptTemplate.from_template(
                template=template,
                input_variable=['context', 'question', 'history'])
    return prompts



def retrieval_chain():
    qa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type='stuff',
        retriever=vector_db,
        chain_type_kwargs={
            'verbose': True,
            #'prompt': prompt(),
            'memory': ConversationBufferMemory(
                memory_key='history',
                input_key='question'
            )
        }
    )
    return qa

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"


def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Selamat datang di MentalChat. Sebelum kita mulai, apakah ada topik atau isu khusus yang ingin Anda diskusikan atau bahas hari ini? Jangan ragu untuk berbagi perasaan, pemikiran, atau pengalaman yang mungkin ingin Anda eksplorasi bersama.")]
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = retrieval_chain()


def get_llm_chain_from_session() -> RetrievalQA:
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
        response = llm_chain.run({'query': prompt})
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
        st.chat_message(ASSISTANT).write(response)