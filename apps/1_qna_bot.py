from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.title('Askai - AI QnA bot')
st.markdown('My QnA Bot with Langchain and Google Gemini')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)

query = st.chat_input('Ask anything')

if query:
    st.session_state.messages.append({
        'role': 'user',
        'content': query
    })
    st.chat_message('user').markdown(query)
    res = llm.invoke(query)
    st.chat_message('ai').markdown(res.content)
    st.session_state.messages.append({
        'role': 'ai',
        'content': res.content
    })

# while True:
#     query = input('User: ')

#     if query.lower() in ['exit', 'quit', 'bye']:
#         print('\nGood Bye\n')
#         break

#     result = llm.invoke(query)
#     print('AI: ',result.content, '\n')

# question = 'Who are the top 10 men in history'

# result = llm.invoke(question)

# print(result.content)