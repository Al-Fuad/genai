from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-20b', streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]

if 'memory' not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt='you are an amazing ai agent and can search on google as well'
)

st.header('Askai - AI QnA bot')

for message in st.session_state.history:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)

query = st.chat_input('Ask anything...')
if query:
    st.chat_message('user').markdown(query)
    st.session_state.history.append({'role': 'user', 'content': query})

    response = agent.stream(
        {'messages': [
            {
                'role': 'user',
                'content': query
            },
        ]},
        {'configurable': {
            'thread_id': '1'
        }},
        stream_mode='messages'
    )

    ai_container = st.chat_message('ai')
    with ai_container:
        space = st.empty()

        message = ''

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)
    # answer = response['messages'][-1].content
    # st.chat_message('ai').markdown(message)
        st.session_state.history.append({'role': 'ai', 'content': message})

