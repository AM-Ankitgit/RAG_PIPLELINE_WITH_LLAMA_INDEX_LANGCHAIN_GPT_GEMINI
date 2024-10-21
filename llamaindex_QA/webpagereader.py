from llama_index.readers import SimpleWebPageReader
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex
import os
import llama_index

from dotenv import load_dotenv
load_dotenv()


def main(url:str)->None:
    
    doc    = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index  = VectorStoreIndex.from_documents(documents=doc)
    query_engine = index.as_query_engine()
    resp = query_engine.query("CI/CD Pipeline")
    print(resp)

if __name__=="__main__":
    main('https://medium.com/@raja.gupta20/generative-ai-for-beginners-part-1-introduction-to-ai-eadb5a71f07d')


