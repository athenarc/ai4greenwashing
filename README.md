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
