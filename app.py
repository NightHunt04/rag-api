from flask import Flask, request, jsonify
from flask_cors import CORS

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

app = Flask(__name__)
CORS(app)

url = ''
docs = ''
vectorStore = ''
chain = ''

def create_vector_db(docs):
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorStore = FAISS.from_documents(docs, embedding = embedding)

    return vectorStore

def get_documents_from_web(url):
    loader = WebBaseLoader(url)

    try:
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        splitDocs = splitter.split_documents(docs)

        return splitDocs
    except Exception as e:
        print(f'Error occured while loading web page: {e}')
        return []
    
def get_documents_from_pdf(path):
    loader = PyPDFLoader(path)

    try:
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        splitDocs = splitter.split_documents(docs)

        return splitDocs    
    except Exception as e:
        print(f'Error occured while loading the pdf: {e}')
        return []

def create_chain(vectorStore):
    llm = ChatGroq(
        model = 'llama-3.1-8b-instant',
        temperature = 0.6,
        max_tokens = 1024,
    )

    # llm = ChatGoogleGenerativeAI(
    #     model='gemini-1.5-flash',
    #     temperature=0.6,
    #     max_output_tokens=1024,
    # )

    prompt = ChatPromptTemplate.from_template('''
    Answer the user's question from the given content, also add some of your knowledge which must be of the same context given in the context. Avoid giving answers to questions which are completely out of context from the given context
    Context: {context}
    Question: {input}
    ''')

    chain = create_stuff_documents_chain(
        llm = llm,
        prompt = prompt,
    )

    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 7})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain


@app.route('/', methods=['GET'])
def check():
    return 'Server is running...'

@app.route('/set-url', methods=['POST'])
def set_url():
    global url, docs, vectorStore, chain

    request_data = request.get_json()
    url = request_data['url']

    docs = get_documents_from_web(url)
    vectorStore = create_vector_db(docs)
    chain = create_chain(vectorStore)
    print(url)
    return 'Url set successfully'

@app.route('/get-answer', methods=['POST'])
def get_answer():
    request_data = request.get_json()
    prompt_ = request_data['prompt']

    response = chain.invoke({ 'input': prompt_ })

    return jsonify({ 'response': response['answer'] })

if __name__ == '__main__':
    app.run()