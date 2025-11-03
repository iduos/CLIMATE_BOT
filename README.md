# Installation

## Clone the repository...

Clone the repository and install dependencies:

```bash
git clone https://github.com/iduos/CLIMATE_BOT.git

cd CLIMATE_BOT
```

## Create a python virtual environment...

```bash
python -m venv bot_env

source bot_env/bin/activate       # linux

.\bot_env\Scripts\Activate.ps1    # windows

python -m pip install --upgrade pip

pip install -r requirements.txt
```

If you get *'Failed to build installable wheels for some pyproject.toml based projects'* with Windows...see APPENDIX at end of this README.md

## Install Ollama and models...

The database included here uses a local model for embedding, nomic-embed-text:latest.

So you will need to install ollama

  https://ollama.com/download

```bash
ollama pull nomic-embed-text:latest

ollama pull qwen3:30b
```

# The files included are...

  **climate_bot.py**                chatbot with selectable viewpoints by positively filtering a vector database

  **build_knowledge_base.py**   scrape, categorise and add post/comments pairs and meta data to a vector database

  /src                      common source files of functions used by the applications

  /vdbs                     vector database

  /json                     some important schema for LLM output verification

  /rubrics                  your json specifying how to categorise comments 

# The chatbot

On linux or a mac you can run the chatbot with local models using ollama 

```bash
  streamlit run climate_bot.py -- \
  --embed_model nomic-embed-text:latest \
  --embed_backend ollama \
  --chat_model gpt-oss:20b \
  --chat_backend ollama \
  --vector_db_path vdbs/climate_uk4_5000GF
```

(For Windows powershell replace \ with ` )

```bash
streamlit run climate_bot.py -- `
  --embed_model nomic-embed-text:latest `
  --embed_backend ollama `
  --chat_model gpt-oss:20b `
  --chat_backend ollama `
  --vector_db_path vdbs/climate_uk4_5000GF
```

You can also run the chatbot with an API key specified in a .env file in the same directory 
as the python files climate_bot.py and build_knowledge_base.py, with the contents, for example 

  GOOGLE_API_KEY=*******your api key********

Hence run with...

```bash
  streamlit run climate_bot.py -- \
  --embed_model nomic-embed-text:latest \
  --embed_backend ollama \
  --embed_model nomic-embed-text:latest \
  --chat_model gemini-2.5-flash \
  --vector_db_path vdbs/climate_uk4_5000GF
```

# Build your own vector database

You will need Reddit credentials for example add to .env

  REDDIT_CLIENT_ID=*****************
  
  REDDIT_CLIENT_SECRET=******************
  
  REDDIT_USER_AGENT=**********************

gemini-2.5-flash is recommended for classification

You can build your own vector database with your own searches and classification criteria.

```bash
  python build_knowledge_base.py process-all \
    --subreddits "worldnews, conservative, liberal, libertarian" \
    --query "climate change OR global warming" \
    --start_date "2025-01-01" \
    --end_date "2025-11-01" \
    --bin_by_period "month" \
    --scrape_limit 1000 \
    --scoring_prompts "rubrics/climateUK4.json" \
    --output "data/environ_category_uk4_1000GF.json" \
    --scorer_model gemini-2.5-flash \
    --scorer_backend gemini \
    --embed_model "nomic-embed-text:latest" \
    --sample_size 1000 \
    --include_justifications \
    --vector_db_path vdbs/your_db_1000GF
```

Replace *rubrics/climateUK4.json** with your file in rubrics folder. Note the json format used.

NOTE gpt-oss:20b or gpt-oss:120b will not work for classification as they don't support the json output needed.


# APPENDIX

If you see this error
× Failed to build installable wheels for some pyproject.toml-based projects
╰─> scikit-learn-extra, hdbscan
error: Microsoft Visual C++ 14.0 or greater is required.

This is a common issue on Windows when installing Python packages that use C/C++ extensions (like scikit-learn-extra or hdbscan).

Fix: Install Microsoft C++ Build Tools
When a pre-compiled wheel isn’t available for your Python version, pip must build the package from source — which requires a C/C++ compiler. On Windows, this compiler is provided by the Microsoft C++ Build Tools.

1. Download the Installer

Get the Visual Studio Community Installer
—or directly the Build Tools.

2. Install the C++ Workload

Run the downloaded installer

In the Workloads tab, check “Desktop development with C++”. Click Install

3. Re-run your install
After installation, restart PowerShell or your Anaconda Prompt to load the new compiler environment, then run:

pip install -r requirements.txt

The packages scikit-learn-extra and hdbscan should now build and install successfully.
