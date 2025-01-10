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