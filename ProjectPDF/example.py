from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("employee_handbook.pdf")



if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH.resolve()}"
    )


# 2. Read every page from the PDF
reader = PdfReader(str(PDF_PATH))

pages = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    if text.strip():
        pages.append(
            Document(
                page_content=text,
                metadata={"page": page_number}
            )
        )

print(f"Pages loaded: {len(pages)}")


# 3. Split the pages into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

print(f"Chunks created: {len(chunks)}")


# 4. Convert the chunks into embeddings
embedding_model = OllamaEmbeddings(
    model="embeddinggemma"
)


# 5. Store the chunks and embeddings
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model
)


# 6. Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# 7. Create the LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# 8. Create the prompt
prompt = ChatPromptTemplate.from_template(
    """
You answer questions using only the supplied context.

If the answer is not available in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = prompt | llm | StrOutputParser()

# 9. Ask a question
question = "How many paid sick days do employees receive?"

relevant_chunks = retriever.invoke(question)


# 10. Combine the retrieved chunks
context = "\n\n".join(
    f"Page {document.metadata.get('page')}:\n"
    f"{document.page_content}"
    for document in relevant_chunks
)


# 11. Send only the relevant information to the LLM
answer = chain.invoke(
    {
        "context": context,
        "question": question
    }
)

print("\nAnswer:")
print(answer)

print("\nPages used:")
for document in relevant_chunks:
    print(document.metadata.get("page"))