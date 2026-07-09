from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import os
import streamlit as st

## data in st session

if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False

if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

def process_document(path):

    ## load doc
    loader = PyPDFDirectoryLoader(path=path)
    docs = loader.load()

    ## splite doc
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splitted_docs = splitter.split_documents(documents=docs)

    ## embeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    ## vector store
    vector_store = InMemoryVectorStore.from_documents(
        documents=splitted_docs,
        embedding=embeddings
    )

    ## tool
    @tool
    def retriever_tool(query:str):
        """
            This tool can help ypu tp retrieve the relevant data of the pdf documents.
        """
        print('Tool called for: ', query)
        docs = vector_store.similarity_search(query=query, k=4)
        context = ''


        for doc in docs:
            context += doc.page_content + '\n\n'

        return context


    ## llm
    llm = ChatGroq(model='openai/gpt-oss-20b')

    ## system prompt
    system_prompt = """
        You are a helpful assistant that answers questions using retrieved context.
        ALWAYS use the 'retriever_tool' tool for for questions requiring external knowledge.
        If you don't know the answer, then you can say that 'I don't know.
    """

    memory = InMemorySaver()

    ## agent
    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

# while True:
#     query = input('User: ')

#     if query.lower() == 'exit':
#         break

#     res = agent.invoke(
#         {
#             'messages': [
#                 {
#                     'role': 'user',
#                     'content': query
#                 }
#             ]
#         },
#         {
#             'configurable': {
#                 'thread_id': 1
#             }
#         }
#     )

#     result = res['messages'][-1].content

#     print('AI: ', result)

## upload ui
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(
        label='Select PDF files',
        type=['pdf', 'docx'],
        accept_multiple_files=True,
    )
    if uploaded:
        with st.spinner('Proccessing...'):
            path = './doc_files/'
            os.makedirs(path, exist_ok=True)
            for file in uploaded:
                filepath = os.path.join(path, file.name)

                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())
            
            process_document(path=path)
            st.rerun()

## chat ui
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get('role')
        content = message.get('content')
        st.chat_message(role).markdown(content)

    query = st.chat_input('Ask anything related to uploaded documents...')
    if query:
        st.session_state.messages.append({'role':'user', 'content': query})

        st.chat_message('user').markdown(query)
        res = st.session_state.agent.invoke(
            {
                'messages': [
                    {
                        'role': 'user',
                        'content': query
                    }
                ]
            },
            {
                'configurable': {'thread_id': 1}
            }
        )

        ans = res['messages'][-1].content

        st.session_state.messages.append({'role':'ai', 'content': ans})

        st.chat_message('ai').markdown(ans)
