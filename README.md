# langchain-chains

This repository contains practice implementations of different **LangChain chains**. Each file demonstrates how to structure and execute various types of chains in LangChain for building flexible AI workflows.  

## 📂 Files Included  

1. **`simple_chain.py`**  
   - A basic chain connecting prompt templates with an LLM.  
   - Demonstrates the core concept of passing inputs → processing → outputs.  

2. **`sequential_chain.py`**  
   - Combines multiple chains in sequence.  
   - The output of one chain is passed as input to the next.  
   - Useful for building step-by-step pipelines.  

3. **`parallel_chain.py`**  
   - Runs multiple chains in parallel.  
   - Collects and aggregates results.  
   - Helpful when different models or prompts should run independently.  

4. **`conditional_chain.py`**  
   - Uses conditions to decide which chain to run.  
   - Demonstrates branching logic based on input.  
   - Useful for dynamic workflows.  

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🛠️ Requirements

Python 3.9+

LangChain

OpenAI or Hugging Face API key (depending on LLM used)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📖 Purpose

This repo is for learning and practicing LangChain chains. Each script highlights a different chaining mechanism to help in building more complex AI agents and applications.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ​ Credits

Special thanks to the **[CampusX YouTube channel](https://www.youtube.com/@campusx-official)** for providing valuable tutorials and guidance that inspired this practice.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 👤 Author

Muqadas Ejaz

BS Computer Science (AI Specialization)

AI/ML Engineer

Data Science & Gen AI Enthusiast

📫 Connect with me on [LinkedIn](https://www.linkedin.com/in/muqadasejaz/)  

🌐 GitHub: [github.com/muqadasejaz](https://github.com/muqadasejaz)
