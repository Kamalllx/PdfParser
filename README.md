# PdfParser

dig deep into
**https://docs.agno.com/knowledge/s3_pdf** (important)
https://docs.agno.com/knowledge/pdf-url
https://docs.agno.com/introduction
https://console.groq.com/docs/agno
https://docs.agno.com/models/groq
https://docs.agno.com/vectordb/introduction
https://docs.agno.com/vectordb/pgvector
https://docs.agno.com/embedder/huggingface
https://docs.agno.com/chunking/document-chunking
https://docs.agno.com/chunking/semantic-chunking
https://docs.agno.com/knowledge/search
https://docs.agno.com/storage/postgres


there r so many more interesting documentations in agno search through them if required but the ones listed above are a must because I wan you to buildthe below application (CLI Bases as fo now):
RAG based PDF Chatbot app , backend python using agno , it has a lot of required documentations, like :
models: groq
VectorDB: Potsgres
Embedding: Huggingface
Chunking: semantic or document chunking (read the ddoc and decide whats best for my usecase)
I felt of /doc based RAG automation was done to  an extent in the "knowledge "URLs which I gave so do check out the :
https://docs.agno.com/knowledge/search
**https://docs.agno.com/knowledge/s3_pdf** (important)
https://docs.agno.com/knowledge/pdf-url

 and other required frame works to make a working terminal based chatbot which takes in the input of any pdf path route and is able to answer it in a perfectly well structured format , the pdfs would be related to physics chemistry etc.... basically pdfs of chapters of ncert class 1 1 and 12 .

this is just a suggestion , please use better methods if you know any : use groq vision model so that it understand the diagrams and images too in the pdf , make it a full fledged app where it uses Postgres vectorDB and after parsing the data of the pdf it create a collection for each pdf parsed or something like that the data is uplaoded to Postgres vectorDB (clustering and classification) ,and the context retrieval for any query asked bt the user should be handled smoothly using vecotr search algorithms /semantic search. 

the chatbot should maintain chat conversation history  / context so that it answers the queries of the client based on the context of conversation.

for now make it a CLI based application.

refer official documentations of whichever libraries and api services r to be used ,I have the agno api_key , so u can use it till the best extent.double check the logic and give full codes of files to be created.
