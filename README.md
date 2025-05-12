

# BeNiFit: Boosting the Analysis of the NFT Market based on Large Language Models


## Overview
This repository contains the code and resources (data) for my ELENE6883 project. The project propose an end-to-end LLM-based pipeline to boost the analysis of the NFT market.

### Team Members
| Name           | Email Address          |
|----------------|----------------|
| Lilin Xu  | lx2331@columbia.edu  | 


## Introduction
Non-Fungible Tokens (NFTs) have experienced rapid growth, making effective market analysis increasingly important. However, the complex nature of the NFT market makes the analysis task challenging. Large Language Models (LLMs) have shown strong capabilities in reasoning and understanding various data and contextual information, making them a promising solution for this task. In this paper, we propose BeNiFit, an end-to-end LLM-based pipeline, to boost the analysis of the NFT market. The pipeline consists of two stages, including NFT data collection and LLM-empowered reasoning. In the stage of NFT data collection, we collect various NFT data through the API provided by OpenSea and organize it into a structured NFT database. In the stage of LLM-empowered analysis, we introduce NFTGPT, a zero-shot method empowered by LLMs' reasoning capabilities and in-context learning. NFTGPT is designed to achieve effective analysis and predictions for NFTs based on both historical trading data and contextual information. Extensive experiments on real-world NFT collections demonstrate that BeNiFit consistently outperforms baselines under various settings, achieving up to a 25% performance improvement.

---

## Results
Experiment results of **BeNiFit** and baselines with GPT-4o are stored under `/results` folder as exmples. 

## How to Run the Project
### 1. Prepare the API keys
This project need four API keys: OpenSea key, OpenAI key, Anthropic key and Google API.

### 2. Code Structure
- **`config` folder**: Stores experiment config files.
- **`data` folder**: data collected from NFT collections.
- **`evaluation` folder**: evaluation codes.
    - **`*.py` end with `api`**: for commercial LLMs.
    - **`*.py` end with `model`**: for open-source LLMs.
- **`promt` folder**: system prompts.
- **`results` folder**: experimental results.
- **`*.ipynb` in main folder**: dataset construction.

### 3. Prepare datasets
#### get information of nfts by collection
1. get_nfts_by_collection.ipynb 
    data/{collection_slug}/raw_response_data.json
    data/{collection_slug}/nfts_by_collection_data.json
2.  json_rearrange.ipynb
    data/{collection_slug}/nfts_by_collection_data_sorted.json
3.  get_nft_trait.ipynb
    data/{collection_slug}/nfts_traits_data_sorted.json
    data/{collection_slug}/images/
4. get_collection_traits.ipynb
    data/{collection_slug}/collection_data.json
    data/{collection_slug}/collection_traits_data.json
    data/{collection_slug}/collection_status_data.json

5.  get_nft_event.ipynb
    data/{collection_slug}/sale_events/
6.  get_collection_event.ipynb
    data/{collection_slug}/collection_sale_events


#### check information
1. use "identifier" to check
    data/{collection_slug}/raw_response_data.json
    data/{collection_slug}/nfts_by_collection_data.json
    data/{collection_slug}/nfts_by_collection_data_sorted.json
    data/{collection_slug}/nfts_traits_data_sorted.json

2. use data_check.ipynb to check
    data/{collection_slug}/images/
    data/{collection_slug}/sale_events/

#### make dataset
data_dataframe_maker.ipynb
    data/{collection_slug}/collection_event_data.csv: records of sale events
    data/{collection_slug}/collection_event_nft_stats.csv: stats of records, groupy by identifier
    data/{collection_slug}/nfts_traits_data_sorted.csv: nft traits
    data/{collection_slug}/nfts_traits_event_data_sorted.csv: nft traits and total sale event numbber

data_collection_analysis.ipynb
    data/{collection_slug}/collection_event_data_date.csv: group all sale events by date

dataset_maker.ipynb
    data/{collection_slug}/dataset/*.csv

evaluation/collection_trading_analysis.py
    data/{collection_slug}/dataset/*.json



### 4. Implementation Example

**BeNiFit**
```bash
cd ./evaluation/
python nft_ours_icl_api.py --config ../config/ours_icl.json --exp ours_icl_gpt-4o_historical_all
python nft_metrics.py --config ../config/ours_icl.json --exp ours_icl_gpt-4o_historical_all
```

**Baselines**
```bash
cd ./evaluation/
python nft_baseline_cot_api.py --config ../config/baseline_cot.json --exp baseline_cot_gpt-4o_historical_all
python nft_metrics.py --config ../config/baseline_cot.json --exp baseline_cot_gpt-4o_historical_all

python nft_baseline_zeroshot_api.py --config ../config/baseline_zeroshot.json --exp baseline_zeroshot_gpt-4o_historical_all
python nft_metrics.py --config ../config/baseline_zeroshot.json --exp baseline_zeroshot_gpt-4o_historical_all
```

---