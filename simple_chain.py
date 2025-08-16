from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain.chains import SimpleChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["input_text"],
    template="Tell me something about {input_text}"
)



def init_model():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
        # mistralai/Mixtral-8x22B-Instruct-v0.1
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm)


model = init_model()

parser= StrOutputParser()


chain = prompt | model | parser

result = chain.invoke({"input_text": "cricket"})


print(result)  # Should print the French translation of "cricket"

chain.get_graph().print_ascii()
