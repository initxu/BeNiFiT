import os
import time
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import requests
from openai import OpenAI
from anthropic import Anthropic
from google import genai
import argparse
from tqdm import tqdm


OPENAI_API_KEY=''


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


class LLMJudger:
    def __init__(self):
        """Initialize the LLM testing framework with API clients."""
        # Load API keys from environment variables
        self.openai_api_key = OPENAI_API_KEY

        # Initialize clients
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
    
    def test_openai(self, prompt: str, model: str = "gpt-4o", temperature: float = 0.7) -> Dict:
        """Test OpenAI models."""
        if not self.openai_api_key:
            return {"error": "OpenAI API key not found in environment variables"}
        
        start_time = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            end_time = time.time()
            
            return {
                "model": model,
                "prompt": prompt,
                "response": response.choices[0].message.content,
                "tokens": response.usage.total_tokens,
                "response_time": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "model": model, "provider": "OpenAI"}
    
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
                
        # Add summary statistics - for stronger baselines
        avg_price = identifier_data['payment_symbol_amount'].mean()
        min_price = identifier_data['payment_symbol_amount'].min()
        max_price = identifier_data['payment_symbol_amount'].max()
        total_sales = len(identifier_data)
        
        data_str += f"Summary: {total_sales} sales, "
        data_str += f"avg: {avg_price:.4f} {symbol}, "
        data_str += f"min: {min_price:.4f} {symbol}, "
        data_str += f"max: {max_price:.4f} {symbol}\n\n"
        return data_str
        # except Exception as e:
        #     return f"Error processing trading data: {str(e)}"
    

def main():
    # Set up argument parser
    
    parser = argparse.ArgumentParser(description='Evaluate prediction and analysis of NFT trading data')
    parser.add_argument('-c', '--config', required=True, 
                       help='Path to config file (JSON format)')
    parser.add_argument('-e', '--exp', required=True, 
                       help='experiment name')
    parser.add_argument('-m', '--model', default='gpt-4o', 
                       help='model as judger')
    parser.add_argument('-p', '--prompt_pth', default='../prompt/metrics/metrics.txt',
                       help='Path to the prompt file') 
    args = parser.parse_args()
    
    # Create your NFT analysis prompt
    exp_config = load_config(args.config)
    exp=exp_config[args.exp]
    method=exp['method']
    collection_slug=exp['collection_slug']
    nft_historical_period=exp['nft_historical_period']
    
    
    results_pth=f'../results/{collection_slug}/{method}/{args.exp}.json'
    historical_nft_data_pth=f'../data/{collection_slug}/dataset/dataset_historical_transactions_{nft_historical_period}.csv'
    # historical_collection_data_pth=f'../data/{collection_slug}/dataset/collection_historical_transactions_{nft_historical_period}.csv'
    collection_trading_info_pth=f'../data/{collection_slug}/dataset/collection_trading_analysis_{nft_historical_period}.json'
    nft_prediction_data_pth=f'../data/{collection_slug}/dataset/dataset_prediction_transactions.csv'
    nfts_traits_pth=f'../data/{collection_slug}/nfts_traits_event_data_sorted.csv'
    
    judge_results_pth=f'../results/metrics/{collection_slug}/{method}/{args.exp}.json'
    
    with open(args.prompt_pth, 'r') as f:   
        metrics_prompt = f.read()
    with open(collection_trading_info_pth, 'r') as f:
        collection_trading_info = json.load(f)['response']
    
    nfts_traits=pd.read_csv(nfts_traits_pth)
    collection_description=nfts_traits['description'][0]
    
    # Initialize the judger
    judger = LLMJudger()
    
    with open(results_pth, 'r') as f:
        results = json.load(f)
        
    judger_results={}
    # for key in results.keys():
    for key in tqdm(results.keys(), desc="Judging results"):
        identifier=int(key.split('-')[1])
        llm_analysis=results[key]['response']
        nft_historical_data=process_nft_trading_data_by_identifier(historical_nft_data_pth, identifier)
        nft_prediction_data=process_nft_trading_data_by_identifier(nft_prediction_data_pth, identifier)
        
        nft_rarity_rank=nfts_traits[nfts_traits['identifier']==identifier]['rarity_rank'].values[0]
        nft_rarity_rank_info="the rarity rank of the NFT in the collection is " + str(nft_rarity_rank)
        
        
        prompt=metrics_prompt.replace("$[Historical Start]$", historical_start_date[nft_historical_period]).replace("$[Historical End]$", historical_end).replace("$[Historical Trading data]$", nft_historical_data).replace("$[April Trading Data]$", nft_prediction_data).replace("$[Rarity information]$", nft_rarity_rank_info).replace("$[Collection Description]$", collection_description).replace("$[Collection Trading Infomation]$", collection_trading_info).replace("$[Test_LLM_Analysis_and_Predictions]$", llm_analysis)
        
        
        
        judger_result=judger.test_openai(prompt, model=args.model,temperature=0)
        judger_results[key] = judger_result
        with open(judge_results_pth, 'w') as f:   
            json.dump(judger_results, f, indent=2)
        
        
        
    #save results
    
    
        
    


if __name__ == "__main__":
    main()