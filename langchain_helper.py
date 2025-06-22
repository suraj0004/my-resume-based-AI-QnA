import os
import time
import logging


from secret import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from langchain.globals import set_debug

set_debug(True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

vectorDB_path = 'faiss_index'

llm = OpenAI(
    model='gpt-4o-mini',
    temperature=0.5,
    api_key=OPENAI_API_KEY,
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def create_vector_db_from_csv():
    logger.debug("Starting to create vector DB from CSV.")
    loader = CSVLoader(file_path='suraj_resume_qna.csv', source_column='Question')
    data = loader.load()
    logger.debug(f"Loaded {len(data)} documents from CSV.")

    vectorDB = FAISS.from_documents(documents=data, embedding=embeddings)
    logger.debug("Created FAISS vector DB from documents.")

    vectorDB.save_local(vectorDB_path)
    logger.debug(f"Saved vector DB locally at {vectorDB_path}.")
    time.sleep(2)


def check_and_create_vector_db():
    if not os.path.exists(vectorDB_path):
        logger.debug(f"Vector DB path '{vectorDB_path}' does not exist. Creating vector DB.")
        create_vector_db_from_csv()
    else:
        logger.debug(f"Vector DB path '{vectorDB_path}' exists. No need to create.")


def ask_question_from_qa_chain(question):
    logger.debug(f"Received question: {question!r}")
    question = question.strip()
    if not question.endswith('?'):
        question += '?'
        logger.debug(f"Appended '?' to question. New question: {question!r}")

    vectorDB = FAISS.load_local(vectorDB_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    logger.debug("Loaded vector DB from local storage.")
    retriever = vectorDB.as_retriever(score_threshold=0.7)

    prompt_template = """
    You are Suraj Upadhyay, an experienced professional. Answer the following question as if you are Suraj himself, using a friendly, clear, and professional tone. Use the information provided in the context to respond accurately. Do not make up answers beyond the given context.

    Context:
    {context}

    Question:
    {question}

    Answer as Suraj:
    """.strip()

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    logger.debug("Initialized RetrievalQA chain.")

    # Get clean result
    result = chain(question).get('result', 'I don’t know').strip()
    logger.debug(f"Raw result from chain: {result!r}")

    # Post-process answer to remove any leading "ANSWER:" or '?'
    if result.lower().startswith("answer:"):
        result = result[7:].strip()
        logger.debug("Stripped leading 'ANSWER:' from result.")

    if result.startswith('?'):
        result = result[1:].strip()
        logger.debug("Stripped leading '?' from result.")

    logger.debug(f"Final processed result: {result!r}")
    return result

