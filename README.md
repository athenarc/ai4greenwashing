# ai4greenwashing

Project for employing AI techniques for detecting and justifying greenwashing claims.

## Installation:

after cloning the repository, it is recommended that you create a virtual enviroment

```bash
# Install Python 3.10 if not already installed
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Create a virtual environment using Python 3.10
python3.10 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Verify the Python version in the virtual environment
python --version

# Execute the following bach commands in the order below:
pip install torch torchvision torchaudio
pip install pip==23.3.1 setuptools==59.5.0 cython==3.0.6 wheel==0.42.0
pip install "deepdoctection[pt]==0.26" --no-deps
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install langchain-ollama chromadb langchain-chroma
pip install tqdm
pip install langchain-groq
```
If something is broken you might need to do:

```bash
pip install "deepdoctection[pt]==0.26" --no-deps
pip install langchain-ollama chromadb langchain-chroma
pip install tqdm
pip install langchain-groq
```
a second time.

Additionally, for the llm API to work, you need to create your own private API key [here](https://console.groq.com/keys), and add it to the .env file that you must create in accordance with the .env.example file.

Your private .env file must also include two large language models that are instaniated on the llm.py file. The Groq llm models that can be inserted can be found [here](https://console.groq.com/docs/models). We recommend the usage of `llama-3.3-70b-versatile` and `llama3-70b-8192` as a secondary one, in case the first one fails. Feel free to experiment with various models, as long as they are supported by the Groq API.

For running locally, install ollama from https://ollama.com/ and then do

```
ollama pull llama3.2
ollama pull mxbai-embed-large
```

This will have worse results due to using a smaller model, but does not need an API. Also, for Linux machines you should use:

```
systemctl disable ollama.service
```

so that it does not start on boot, and start/stop ollama using:

```
systemctl start/stop ollama.service
```

otherwise the process will keep restarting even if killed.

## Running:

Run either:
```bash
python -m reportparse.main \
  -i ./reportparse/asset/example.pdf \
  -o ./results \
  --input_type "pdf" \
  --overwrite_strategy "no" \
  --reader "pymupdf" \
  --annotators "llm"
```
or to use layout analysis: 

```bash
python -m reportparse.main \
  -i ./reportparse/asset/example.pdf \
  -o ./results \
  --input_type "pdf" \
  --overwrite_strategy "no" \
  --reader "deepdoctection" \
  --annotators "llm"
  ```

You can use ollama_llm instead of llm to run locally.
This will store all pages in a chromadb and afterwards annotate each page.

Then run:
```bash
python -m reportparse.adjudicator
```
for final verdicts.