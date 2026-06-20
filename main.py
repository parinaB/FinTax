from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os


load_dotenv()

docs=[]
pdf_folder="documents"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        try:
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed: {file}")
            print(e)
        
        
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)

chunks=splitter.split_documents(docs)
# print(chunks[0].page_content)

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts=[]

for chunk in chunks:
    texts.append(chunk.page_content)
    
    
embeds=embeddings.embed_documents(texts)
print(len(embeds[0])) 


if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )
    vectorstore.save_local("faiss_index")



    
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        
    }
)




template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are FinTax AI, a specialized assistant for Indian taxation, accounting, and finance.

        Answer questions strictly using the information provided in the retrieved context.
        Do not rely on outside knowledge or make assumptions.

        Guidelines:
        - If the answer is not available in the retrieved documents, clearly state that the information could not be found.
        - Never fabricate GST rates, tax slabs, deductions, TDS percentages, or financial figures.
        - Provide concise yet accurate explanations.
        - Include step-by-step calculations whenever applicable.
        - Mention the source document and page reference when available.
        - Highlight any compliance requirements, limitations, or potential risks relevant to the query.
        - Present monetary amounts in Indian Rupees (₹).

        Prioritize factual accuracy, transparency, and source-backed responses.
        """
    ),
    ("human", "{data}")
])


llm = ChatMistralAI(model="mistral-small-latest")
while True: 
    query = input("\nAsk FinTax AI: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
llm = ChatMistralAI(model="mistral-small-latest")
while True: 
    query = input("\nAsk FinTax AI: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    finds=retriever.invoke(query)

# for i, doc in enumerate(finds, start=1):
#     print(f"\nChunk {i}")
#     print(doc.page_content[:500])


    context = "\n\n".join(
        [find.page_content for find in finds]
    )

    prompt=template.format_messages(
        data=f"""
        Context : {context}
    
        Question: {query}
        """
    )

    response=llm.invoke(prompt)
    print(response.content)
    finds=retriever.invoke(query)

# for i, doc in enumerate(finds, start=1):
#     print(f"\nChunk {i}")
#     print(doc.page_content[:500])


    context = "\n\n".join(
        [find.page_content for find in finds]
    )

    prompt=template.format_messages(
        data=f"""
        Context : {context}
    
        Question: {query}
        """
    )

    response=llm.invoke(prompt)
    print(response.content)

