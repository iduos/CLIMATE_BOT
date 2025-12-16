# About

Code to implement virtual humans whose responses to environmental messaging are shaped by their media diets and social interactions. The project scraped thousands of social media post–comment pairs related to environmental issues, classified them by viewpoint through the large-scale orchestration of multiple instances of large language models, and built a vector database of embedded interactions with associated classification metadata to serve as a knowledge source for a chatbot. Dynamic, metadata-based filtering of this knowledge source, in conjunction with retrieval-augmented generation, enabled a chatbot with selectable personas that generate responses to new social media posts based on stereotypical viewpoints grounded in current news, attitudes and zeitgeists.

This repository does not include any Reddit-derived text, embeddings, or persisted vector databases. Although the data analysed in this project were sourced from publicly accessible Reddit posts and comments, Reddit’s terms of service restrict the redistribution of derived datasets, including embedded representations of user-generated content.

To support transparency and reproducibility, this repository provides the full codebase for data collection, embedding, clustering, and vector-database construction, along with detailed instructions for reconstructing the dataset locally using Reddit post and comment identifiers and publicly available APIs.

# Installation

## Clone the repository...

Clone the repository and install dependencies:

```bash
git clone https://github.com/iduos/CLIMATE_BOT.git

cd CLIMATE_BOT
```

## Create a python virtual environment...

```bash
python -m venv climate_env

source climate_env/bin/activate       # linux

.\climate_env\Scripts\Activate.ps1    # windows

python -m pip install --upgrade pip

pip install -r requirements.txt
```


# The files included are...


  **build_knowledge_base.py**   scrape, categorise and add post/comments pairs and meta data to a vector database

  **/rubrics**                  your json specifying how to categorise comments 

  **climate_bot.py**            chatbot with selectable viewpoints by positively filtering a vector database
  
  **/src**                      common source files of functions used by the applications

  **/json**                     some important schema for LLM output verification



# Build your own vector database

You will need Reddit credentials for example add to **.env**

  REDDIT_CLIENT_ID=*****************
  
  REDDIT_CLIENT_SECRET=******************
  
  REDDIT_USER_AGENT=**********************

  GOOGLE_API_KEY=*******your api key********

The cloud based model gemini-2.5-flash is recommended for classification, and gemini-embedding-001 for embedding.

If you prefer local models you will need to install ollama

  https://ollama.com/download

qwen3:30b or llama3.1 models work though results less consistent.

```bash
ollama pull nomic-embed-text:latest

ollama pull qwen3:30b
```

You can build your own vector database with your own searches and classification criteria.

## Linux (local model example)

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
    --scorer_model qwen3:30b \
    --scorer_backend ollama \
    --embed_model "nomic-embed-text:latest" \
    --embed_backend ollam \
    --sample_size 1000 \
    --include_justifications \
    --vector_db_path vdbs/your_db_1000
```

## Windows PowerShell (cloud based model example)

```bash
python build_knowledge_base.py process-all `
    --subreddits "worldnews, conservative, liberal, libertarian" `
    --query "climate change OR global warming" `
    --start_date "2025-01-01" `
    --end_date "2025-11-01" `
    --bin_by_period "month" `
    --scrape_limit 1000 `
    --scoring_prompts "rubrics/climateUK4.json" `
    --output "data/environ_category_uk4_1000GF.json" `
    --scorer_model gemini-2.5-flash `
    --scorer_backend gemini `
    --embed_model gemini-embedding-001 `
    --embed_backend gemini `
    --sample_size 10 `
    --include_justifications `
    --vector_db_path vdbs/your_db_1000
```


If you want to try your own categories, you can replace **rubrics** / **climateUK4.json** with your file in rubrics folder. Note the json format used, replace feature name, options and rubric

```json
[
  {
    "feature": "Climate Attitude UK",
    "type": "category",
    "options": [
      "Concerned",
      "Paradoxical",
      "Sceptical",
      "Irrelevant"
    ],
    "rubric": [
      "Classify into one of four clusters based on expressed viewpoints Concerned, Paradoxical, Sceptical, Irrelevant",
      " ",
      "Sceptical:",
      "The comment downplays the risks of climate change.",
      "And/or the comment suggests impacts are exaggerated, distant, or not urgent.",
      "And/or the comment expresses low support for strong action.",
      "Corresponds to \"Doubtful\" or \"Denial\" groups in earlier research.",
      " ",
      "Concerned:",
      "The comment acknowledges that climate change is real, imminent, and not exaggerated.",
      "And/or the comment expresses that urgent action is needed to address risks.",
      "And/or the comment strongly agrees that people will be affected within 30 years and that the crisis is not beyond control.",
      " ",
      "Paradoxical:",
      "The comment holds mixed or ambivalent views.",
      "And/or the comment sometimes agrees with sceptical statements (e.g., crisis exaggerated), but also believes impacts are coming soon.",
      "And/or the comment expresses powerlessness, believing risks are real but out of human control.",
      " ",
      "Irrelevant:",
      "No relevant content to classify.",
      "And/or the comment nonsensical or too off-topic."
    ],
    "extra_instructions": [
      "When a comment uses strong negative, sarcastic, or aggressive language, determine the *target* of the tone.",
      "If the critique is directed at specific *policies, politicians, or other groups* (e.g., 'morons,' 'a joke') ", 
      "do not automatically classify it as 'Sceptical', this can be Sceptical, but also expression of frustration ", 
      "or cynical 'Concerned' or 'Paradoxical' views.",
      "For lengthy or complex comments, weigh the *core argument* and majority sentiment over single qualifying sentences."
    ]
  }
]
```

NOTE gpt-oss:20b or gpt-oss:120b will not work for classification as they don't support the json output needed.

# The chatbot

On linux or a mac you can run the chatbot with 

```bash
  streamlit run climate_bot.py -- \
  --embed_model nomic-embed-text:latest \
  --embed_backend ollama \
  --chat_model qwen3:30b \
  --chat_backend ollama \
  --vector_db_path vdbs/your_db_1000
```

(For Windows powershell replace \ with ` )

```bash
streamlit run climate_bot.py -- `
  --embed_model gemini-embedding-001 `
  --embed_backend gemini `
  --chat_model qwen3:30b `
  --chat_model gemini-2.5-flash `
  --chat_backend gemini `
  --vector_db_path vdbs/your_db_1000
```

You can also run the chatbot with local ollama models or with an API key specified in a .env file in the same directory as the python files climate_bot.py and build_knowledge_base.py.

