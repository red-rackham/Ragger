# Verify your Ollama-backed LLM is working with LangChain

from langchain_ollama.llms import OllamaLLM

# Initialize the OllamaLLM wrapper with the 'mistral' model you pulled
llm = OllamaLLM(model="mistral")

# Generate a simple test completion
response = llm.generate(["Compile a list of ten stupid things to do"])

# Print out the first generated text
print(response.generations[0][0].text)




