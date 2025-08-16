from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain.chains import SimpleChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 =HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational",   # ✅ not text-generation
    provider="together"
)

model2  = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b",
    task="text-generation",  
    provider="together"
)


prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Streamlit is an open-source Python library for building interactive web apps using only Python. It's ideal for creating dashboards, data-driven web apps, reporting tools and interactive user interfaces without needing HTML, CSS or JavaScript.

This article introduces key Streamlit features, shows how to build a simple app and explains how to run it on a local server using minimal code.

Streamlit Installation
Make sure Python and pip are already installed on your system. To install Streamlit, run the following command in the command prompt or terminal:

pip install streamlit
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()

