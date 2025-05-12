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
from tqdm import tqdm 
import argparse

OPENAI_API_KEY=' '
ANTHROPIC_API_KEY=" "
GOOGLE_API_KEY=" "

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


class LLMTester:
    def __init__(self):
        """Initialize the LLM testing framework with API clients."""
        # Load API keys from environment variables
        self.openai_api_key = OPENAI_API_KEY
        self.anthropic_api_key = ANTHROPIC_API_KEY
        self.google_api_key = GOOGLE_API_KEY
        
        # Initialize clients
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        if self.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
        if self.google_api_key:
            self.gemini_client=genai.Client(api_key=self.google_api_key)
    
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
    
    def test_anthropic(self, prompt: str, model: str = "claude-3-sonnet-20240229", temperature: float = 0.7) -> Dict:
        """Test Anthropic's Claude models."""
        if not self.anthropic_api_key:
            return {"error": "Anthropic API key not found in environment variables"}
        
        start_time = time.time()
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=8000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            end_time = time.time()
            
            return {
                "model": model,
                "prompt": prompt,
                "response": response.content[0].text,
                "tokens": response.usage.input_tokens + response.usage.output_tokens,
                "response_time": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "model": model, "provider": "Anthropic"}
    
    def test_gemini(self, prompt: str, model: str = "gemini-2.0-flash", temperature: float = 0.7) -> Dict:
        """Test Google's Gemini models."""
        if not self.google_api_key:
            return {"error": "Google API key not found in environment variables"}
        
        start_time = time.time()
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            end_time = time.time()
            
            response = self.gemini_client.models.generate_content(
            model=model, contents=prompt
            )
            
            return {
                "model": model,
                "prompt": prompt,
                "response": response.text,
                "tokens": "N/A",  # Google doesn't provide token count in same way
                "response_time": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "model": model, "provider": "Google"}
    



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
    model=exp['model']
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
    # Initialize the tester
    tester = LLMTester()

    for identifier in tqdm(identifier_ls):

        print(f"Processing NFT: {identifier}")
        nft_historical_data=process_nft_trading_data_by_identifier(historical_nft_data_pth, identifier)
        
        prompt=nft_prompt.replace("$[Historical Trading Data]$", nft_historical_data).replace("$[Historical Start]$", historical_start_date[nft_historical_period]).replace("$[Historical End]$", historical_end)


        if model in ['gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18', 'gpt-4.1-2025-04-14', 'gpt-3.5-turbo-0125']:
            nft_result=tester.test_openai(prompt, model=model)
        elif model in ['claude-3-7-sonnet-20250219', 'claude-3-5-haiku-20241022']:
            nft_result=tester.test_anthropic(prompt, model=model)
        elif model in ['gemini-1.5-flash']:
            nft_result=tester.test_gemini(prompt, model=model)
        
        results[f"nft-{identifier}"] = nft_result
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
   
if __name__ == "__main__":
    main()