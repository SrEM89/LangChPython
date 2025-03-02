# Retrieval Chain : load the document from text.
# 1. Create a document from text
# 2. Create and invoke the document with the chain
# 3. Create and invoke the document with create_stuff_documents_chain


from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
#from langchain.chains.combine_documents import create_stuff_documents_chain

docA = Document(
    page_content= "LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production). To highlight a few of the reasons you might want to use LCEL"
)

# Initialize the ChatGroq object
llm = ChatGroq(
    #model="llama-3.2-3b-preview",
    model ="llama-3.3-70b-versatile",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(""" 
    Answer the user's question:
    Context :  {context}
    User Question : {input}
""")

chain = prompt | llm
response = chain.invoke({
    'input':"What is LCEL",
    'context': [docA]
})
print(response.content)


# chain = create_stuff_documents_chain(
#     llm =llm,
#     prompt = prompt
#     )

# response = chain.invoke({
#     'input':"What is LCEL",
#     'context': [docA]
# })
# print(response)