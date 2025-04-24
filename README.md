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
pip install langchain-ollama chromadb langchain-groq
pip install sentence-transformers
```

You will also need to install the Crawl4AI tool in order to run the web_crawler annotator. Instructions can be found [here](https://github.com/unclecode/crawl4ai). You should install it inside your virtual environment.

Additionally, for the llm API to work, you need to create your own private API key [here](https://console.groq.com/keys), and add it to the .env file that you must create in accordance with the .env.example file.

Our databases that are utilized for the various RAG pipelines, are located in the ./reportparse/database_data folder. This folder is a prerequisite for the project to run. The folder has not been uploaded to the GitHub repository because of its size constraints. In case need access to it, feel free to contact us.

Your private .env file must also include two large language models that are instaniated on the llm.py file. The Groq llm models that can be inserted can be found [here](https://console.groq.com/docs/models). We recommend the usage of `llama-3.3-70b-versatile` and `llama3-70b-8192` as a secondary one, in case the first one fails. Feel free to experiment with various models, as long as they are supported by the Groq API.

## Running:

Run:

```bash
python -m reportparse.main   -i reportparse/asset/example.pdf   -o ./results   --input_type "pdf"   --overwrite_strategy "all" --max_pages 10  --reader "pymupdf"   --annotators "llm_agg" --pages_to_gw 1
```

The above command runs the web_rag and chroma_db implementations, to parse the esg-report for greenwashing, from page 1 to a maximum page number defined by the `--pages_to_gw` parameter. The `--max_pages` parameter defines the total number of pages that the chroma_db will store on its database. In our case, `--max_pages 10` means that the chroma db will store the first 10 pages of the esg-report. You can check the results of the implementation on the
`.results/example.pdf.json` file. For more information on how the above bash command works, please refer to this [README.md file](https://github.com/climate-nlp/reportparse/blob/dev/README.md)

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
