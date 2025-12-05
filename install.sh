# Description: Install the necessary dependencies to run the project
python -m venv eval_env
source eval_env/bin/activate
pip install -r requirements.txt

# Create the necessary directories
mkdir cache
mkdir generated_files
mkdir logs_similarity
mkdir logs_surprisal
mkdir texts
mkdir datasets
mkdir results_excel

# Download datasets outside HuggingFace
wget https://raw.githubusercontent.com/proxectonos/calame-gl/main/2025-01-22_calame_rev.json -P datasets