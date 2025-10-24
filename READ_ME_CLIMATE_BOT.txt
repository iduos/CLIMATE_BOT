The files included are...

  climate_bot.py                chatbot with selectable viewpoints by positively filtering a vector database

  build_knowledge_base.py   scrape, categorise and add post/comments pairs and meta data to a vector database

  /src                      common source files of functions used by the applications

  /vdbs                     vector database

  /json                     some important schema for LLM output verification

  /rubrics                  your json specifying how to categorise comments 

To install and use, create a venv or conda environment

  conda create --name climate_bot

  conda activate climate_bot

  pip install -r requirements.txt

The database included here uses a local model for embedding, nomic-embed-text:latest.

So you will need to install ollama

  https://ollama.com/download

  ollama pull nomic-embed-text:latest


One linux or a mac you can run the chatbot with local models using ollama 

  streamlit run climate_bot.py -- \
  --embed_model nomic-embed-text:latest \
  --embed_backend ollama \
  --chat_model gpt-oss:20b \
  --chat_backend ollama \
  --vector_db_path vdbs/climate_uk4_5000GF

(For Windows powershell replace \ with ` )

You can also run the chatbot with an API key specified in a .env file in the same directory 
as the python files climate_bot.py and build_knowledge_base.py, with the contents, for example 

  GOOGLE_API_KEY=*******your api key********

Hence run with...

  streamlit run climate_bot.py -- \
  --embed_model nomic-embed-text:latest \
  --embed_backend ollama \
  --embed_model nomic-embed-text:latest \
  --chat_model gemini-2.5-flash \
  --vector_db_path vdbs/climate_uk4_5000GF


You can build your own vector database with your own searches and classification criteria.

You will need Reddit credentials for example add to .env

  REDDIT_CLIENT_ID=*****************
  REDDIT_CLIENT_SECRET=******************
  REDDIT_USER_AGENT=**********************

gemini-2.5-flash is recommended for classification

  python build_knowledge_base.py process-all \
    --subreddits "worldnews, conservative, liberal, libertarian" \
    --query "climate change OR global warming" \
    --start_date "2025-01-01" \
    --end_date "2025-11-01" \
    --bin_by_period "month" \
    --scrape_limit 1000 \
    --scoring_prompts "rubrics/climateUK4.json" \
    --output "data/environ_category_uk4_100GF.json" \
    --scorer_model gemini-2.5-flash \
    --scorer_backend gemini \
    --embed_model "nomic-embed-text:latest" \
    --sample_size 1000 \
    --include_justifications \
    --vector_db_path vdbs/your_db_1000GF

NOTE gpt-oss:20b or gpt-oss:120b will not work for classification as they don't support the json output needed.


python build_knowledge_base.py process-all \
    --subreddits "worldnews, conservative, liberal, libertarian" \
    --query "climate change OR global warming" \
    --start_date "2025-01-01" \
    --end_date "2025-11-01" \
    --bin_by_period "month" \
    --scrape_limit 1000 \
    --scoring_prompts "rubrics/climateUK4.json" \
    --output "data/environ_category_uk4_100GF.json" \
    --scorer_model magistral:24b \
    --scorer_backend ollama \
    --embed_model "nomic-embed-text:latest" \
    --embed_backend ollama \
    --sample_size 1000 \
    --include_justifications \
    --vector_db_path vdbs/your_db_1000M