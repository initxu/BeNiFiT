import os
import time
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import requests
from tqdm import tqdm 
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer



historical_start_date={
    'all': '2021-07-22',
    '1m': '2025-03-01',
    '3m': '2025-01-01',
    '6m': '2024-10-01',
} 
historical_end = '2025-03-31'


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config



def process_nft_trading_data_by_identifier(csv_path: str, target_identifier: int = None) -> str:
        """Process trading data grouped by identifier from CSV file."""
        # try:

        df_sales = pd.read_csv(csv_path)
        
        # Convert dates for sorting - handle ISO 8601 format with timezone
        df_sales['event_timestamp_iso'] = pd.to_datetime(
            df_sales['event_timestamp_iso'].str.replace('Z', ''),
            utc=True
        )
        
        
        # If target_identifier is specified, filter for that specific NFT
        df_sales = df_sales[df_sales['identifier'] == target_identifier]
        data_str = ""
        

        identifier_data = df_sales[df_sales['identifier'] == target_identifier].sort_values('closing_date_iso')

                
        trading_records = []
        for idx, row in identifier_data.iterrows():
            # Use the timestamp if closing_date_iso is empty or use event_timestamp_iso as fallback
            date_field = row['event_timestamp_iso']
            date = date_field.strftime('%Y-%m-%d %H:%M')
            price = float(row['payment_symbol_amount'])
            symbol = row['payment_symbol']
            # trading_records.append(f"- Date: {date}, Price: {price} {symbol}")
            trading_records.append(f"- {date}: {price} {symbol} ")
        
        # data_str += "data: [" + ", ".join(trading_records) + "]\n"
        data_str += "".join(trading_records) + "\n"
                

        return data_str


def main():
    # Set up argument parser
    
    parser = argparse.ArgumentParser(description='Process NFT data')
    parser.add_argument('-c', '--config', required=True, 
                       help='Path to config file (JSON format)')
    parser.add_argument('-exp', '--exp', required=True, 
                       help='experiment name')
    args = parser.parse_args()
    
    # Create your NFT analysis prompt
    exp_config = load_config(args.config)
    exp=exp_config[args.exp]
    method=exp['method']
    model=exp['model'] # "Qwen/Qwen2.5-3B-Instruct"
    collection_slug=exp['collection_slug']
    nft_historical_period=exp['nft_historical_period']
    
    nft_prompt_pth=f'../prompt/evaluation/{method}.txt'
    historical_nft_data_pth=f'../data/{collection_slug}/dataset/dataset_historical_transactions_{nft_historical_period}.csv'
    dataset_stats_pth=f'../data/{collection_slug}/dataset/dataset_stats.csv'
    output_file=f'../results/{collection_slug}/{method}/{args.exp}.json'
    
    dataset_stats=pd.read_csv(dataset_stats_pth)
    identifier_ls=dataset_stats['identifier'].tolist()
    results = {}
    
    
    with open(nft_prompt_pth, 'r') as f:   
        nft_prompt = f.read()
        
    # Load the model and tokenizer
    local_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model)    
    

    for identifier in tqdm(identifier_ls):

        print(f"Processing NFT: {identifier}")
        nft_historical_data=process_nft_trading_data_by_identifier(historical_nft_data_pth, identifier)
        
        prompt=nft_prompt.replace("$[Historical Trading Data]$", nft_historical_data).replace("$[Historical Start]$", historical_start_date[nft_historical_period]).replace("$[Historical End]$", historical_end)

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(local_model.device)
        start_time = time.time()
        generated_ids = local_model.generate(
            **model_inputs,
            max_new_tokens=8192
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end_time = time.time()
        nft_result={
            "model": model,
            "prompt": prompt,
            "response": response,
            "tokens": "N/A",  
            "response_time": end_time - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        results[f"nft-{identifier}"] = nft_result
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()