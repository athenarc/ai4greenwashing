# ai4greenwashing

Project for employing AI techniques for detecting and justifying greenwashing claims.

## How to install and run:

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
```

Additionally, for the llm API to work, you need to create your own private keys [here](https://console.groq.com/keys), and add them to the .env file that you must create in accordance with the .env.example file.

Your private .env file must also include two large language models that are instaniated on the llm.py file. The Groq llm models that can be inserted to the .env file can be found [here](https://console.groq.com/docs/models). We recommend the usage of `llama-3.3-70b-versatile` and `llama3-70b-8192` as a secondary one, in case the first one fails. Feel free to experiment with various models, as long as they are supported by the Groq API.

In order to run the pdf parser with the llm feature enabled, via terminal, on a page-wise approach, execute the following command:

```bash
python -m reportparse.main   -i ./reportparse/asset/example.pdf   -o ./results   --input_type "pdf"   --overwrite_strategy "all" --max_pages 5  --reader "pymupdf"   --annotators "llm" --llm_text_level page
```

For more information on how to run the pdf parser with different set of features, please refer to the following's repository [README.md](https://github.com/climate-nlp/reportparse/tree/main)
