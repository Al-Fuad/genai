from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

load_dotenv()

db = SQLDatabase.from_uri('sqlite:///my_task.db')
db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK (status IN ('pending', 'in_progess', 'completed')) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
       )
""")

model = ChatOpenAI(model='gpt-5-nano-2025-08-07', streaming=True)
memory = MemorySaver()
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

system_prompt = '''
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of task, present the output in a structured table format to ensure a clean and organized display in the browser.

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description status(pending/in_progress/completed), created_at.
'''

@st.cache_resource
def get_agent():
    agent = create_agent(
        model=model,
        checkpointer=memory,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent

agent = get_agent()

st.subheader('Taskai - Manage your tasks')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Ask me to manage your tasks')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
# while True:
#     query = input('User: ')
#     if query.lower() in ['bye', 'exit', 'quit']:
#         print('Good Bye')
#         break
    with st.chat_message('ai'):
        with st.spinner('Processing...'):
            response = agent.stream(
                {
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                },
                {
                    'configurable': {
                        'thread_id': '1'
                    }
                },
                stream_mode='messages'
            )

            space = st.empty()

            message = ''

            for chunk in response:
                message = message + chunk[0].content
                space.write(message)
            # result = response['messages'][-1].content
            # st.markdown(result)
            st.session_state.messages.append({'role':'ai', 'content':message})
    # print('AI: ', result)