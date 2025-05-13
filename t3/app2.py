# Auto-generated from notebook4a08ae599b.ipynb



from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import textwrap
import time

import langchain

### loaders
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

### splits
from langchain.text_splitter import RecursiveCharacterTextSplitter

### prompts
from langchain import PromptTemplate, LLMChain

### vector stores
from langchain.vectorstores import FAISS

### models
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings

### retrievers
from langchain.chains import RetrievalQA

import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

clear_output()

# --- Cell ---
print('langchain:', langchain.__version__)
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)

# --- Cell ---
sorted(glob.glob('/kaggle/input/100-llm-papers-to-explore/*'))

# --- Cell ---
class CFG:
    # LLMs
    model_name = 'llama2-13b-chat' # wizardlm, llama2-7b-chat, llama2-13b-chat, mistral-7B
    temperature = 0
    top_p = 0.95
    repetition_penalty = 1.15    

    # splitting
    split_chunk_size = 800
    split_overlap = 0
    
    # embeddings
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'    

    # similar passages
    k = 6
    
    # paths
    PDFs_path = '/kaggle/input/100-llm-papers-to-explore'
    Embeddings_path =  '/kaggle/input/faiss-hp-sentence-transformers'
    Output_folder = './rag-vectordb'

# --- Cell ---
def get_model(model = CFG.model_name):

    print('\nDownloading model: ', model, '\n\n')

    if model == 'wizardlm':
        model_repo = 'TheBloke/wizardLM-7B-HF'
        
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
        )        

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config = bnb_config,
            device_map = 'auto',
            low_cpu_mem_usage = True
        )
        
        max_len = 1024

    elif model == 'llama2-7b-chat':
        model_repo = 'daryl149/llama-2-7b-chat-hf'
        
        tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config = bnb_config,
            device_map = 'auto',
            low_cpu_mem_usage = True,
            trust_remote_code = True
        )
        
        max_len = 2048

    elif model == 'llama2-13b-chat':
        model_repo = 'daryl149/llama-2-13b-chat-hf'
        
        tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
        )
                
        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config = bnb_config,       
            device_map = 'auto',
            low_cpu_mem_usage = True,
            trust_remote_code = True
        )
        
        max_len = 2048 # 8192

    elif model == 'mistral-7B':
        model_repo = 'mistralai/Mistral-7B-v0.1'
        
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
        )        

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config = bnb_config,
            device_map = 'auto',
            low_cpu_mem_usage = True,
        )
        
        max_len = 1024

    else:
        print("Not implemented model (tokenizer and backbone)")

    return tokenizer, model, max_len

# --- Cell ---
%%time

tokenizer, model, max_len = get_model(model = CFG.model_name)

clear_output()

# --- Cell ---
model.eval()

# --- Cell ---
### check how Accelerate split the model across the available devices (GPUs)
model.hf_device_map

# --- Cell ---
### hugging face pipeline
pipe = pipeline(
    task = "text-generation",
    model = model,
    tokenizer = tokenizer,
    pad_token_id = tokenizer.eos_token_id,
#     do_sample = True,
    max_length = max_len,
    temperature = CFG.temperature,
    top_p = CFG.top_p,
    repetition_penalty = CFG.repetition_penalty
)

### langchain pipeline
llm = HuggingFacePipeline(pipeline = pipe)

# --- Cell ---
llm

# --- Cell ---
%%time
### testing model, not using the 100 research papers yet
### answer is not necessarily related to those papers
query = "Give me 5 examples of cool gen ai's and explain what they do"
llm.invoke(query)

# --- Cell ---
CFG.model_name

# --- Cell ---
%%time

loader = DirectoryLoader(
    CFG.PDFs_path,
    glob="./*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)

documents = loader.load()

# --- Cell ---
print(f'We have {len(documents)} pages in total')

# --- Cell ---
documents[8].page_content

# --- Cell ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CFG.split_chunk_size,
    chunk_overlap = CFG.split_overlap
)

texts = text_splitter.split_documents(documents)

print(f'We have created {len(texts)} chunks from {len(documents)} pages')

# --- Cell ---
%%time
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

vectordb = FAISS.from_documents(
    texts,
    HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
)

### persist vector database
vectordb.save_local(f"{CFG.Output_folder}/faiss_index_rag") # save in output folder
#     vectordb.save_local(f"{CFG.Embeddings_path}/faiss_index_hp") # save in input folder

clear_output()

# --- Cell ---
### test if vector DB was loaded correctly
vectordb.similarity_search('BERT')

# --- Cell ---
prompt_template = """
Don't try to make up an answer, if you don't know just say that you don't know.
Answer in the same language the question was asked.
Use only the following pieces of context to answer the question at the end.

{context}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["context", "question"]
)

# --- Cell ---
# llm_chain = LLMChain(prompt=PROMPT, llm=llm)
# llm_chain

# --- Cell ---
retriever = vectordb.as_retriever(search_kwargs = {"k": CFG.k, "search_type" : "similarity"})

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever, 
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)

# --- Cell ---
### testing MMR search
question = "Give me brief info on Megatron-LM"
vectordb.max_marginal_relevance_search(question, k = CFG.k)

# --- Cell ---
### testing similarity search
question = "Give me brief info on Megatron-LM"
vectordb.similarity_search(question, k = CFG.k)

# --- Cell ---
def wrap_text_preserve_newlines(text, width=700):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])
    
    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4]
            + ' - page: '
            + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )
    
    ans = ans + '\n\nSources: \n' + sources_used
    return ans

# --- Cell ---
def llm_ans(query):
    start = time.time()
    
    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)
    
    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str

# --- Cell ---
CFG.model_name

# --- Cell ---
query = "List all the layers present in BERT model"
print(llm_ans(query))

# --- Cell ---
query = "What is GenAI and what are it's advantages?"
print(llm_ans(query))

# --- Cell ---
query = "What are horcrux?"
print(llm_ans(query))

# --- Cell ---
query = "Give me 5 examples of cool gen ai and explain what they do"
print(llm_ans(query))

# --- Cell ---
# import locale
# locale.getpreferredencoding = lambda: "UTF-8"

# --- Cell ---
# ! pip install --upgrade gradio -qq
# clear_output()

# --- Cell ---
# import gradio as gr
# print(gr.__version__)

# --- Cell ---
# def predict(message, history):
#     # output = message # debug mode

#     output = str(llm_ans(message)).replace("\n", "<br/>")
#     return output

# demo = gr.ChatInterface(
#     predict,
#     title = f' Open-Source LLM ({CFG.model_name}) for Harry Potter Question Answering'
# )

# demo.queue()
# demo.launch()

# --- Cell ---


