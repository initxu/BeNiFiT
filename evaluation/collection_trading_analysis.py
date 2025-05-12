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

OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GOOGLE_API_KEY=""

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
    
    

def process_collection_trading_data_by_identifier(csv_path: str) -> str:
        """Process trading data grouped by identifier from CSV file."""
        # try:

        df_sales = pd.read_csv(csv_path)
        data_str = ""
        # Convert dates for sorting - handle ISO 8601 format with timezone
        df_sales['date'] = pd.to_datetime(
            df_sales['date'],
            utc=True
        )

                
        trading_records = []
        for idx, row in df_sales.iterrows():
            # Use the timestamp if closing_date_iso is empty or use event_timestamp_iso as fallback
            date_field = row['date']
            date = date_field.strftime('%Y-%m-%d')
            price = float(row['floor_price_eth'])
            volume= float(row['total_volume'])
            trade_count = int(row['trade_count'])
            symbol = 'ETH'
            # trading_records.append(f"- Date: {date}, Price: {price} {symbol}")
            trading_records.append(f"- {date}: {price} {symbol}, {volume} {symbol} ")
        
        # data_str += "data: [" + ", ".join(trading_records) + "]\n"
        data_str += "".join(trading_records) + "\n"

        return data_str


def main():
    # Set up argument parser
    
    parser = argparse.ArgumentParser(description='Process NFT data')
    parser.add_argument('-cs', '--collection_slug', required=True, 
                       help='collection slug')
    parser.add_argument('-hp', '--historical_period', required=True, 
                       help='historical period')
    parser.add_argument('-m', '--model', required=True, 
                       help='LLM model for analysis')
    parser.add_argument('-p', '--prompt', required=True, 
                       help='prompt path for analysis')
    args = parser.parse_args()
    

    collection_slug=args.collection_slug
    historical_period=args.historical_period
    
    
    historical_collection_data_pth=f'../data/{collection_slug}/dataset/collection_historical_transactions_{historical_period}.csv'
    nfts_traits_pth=f'../data/{collection_slug}/nfts_traits_event_data_sorted.csv'

    output_file=f'../data/{collection_slug}/dataset/collection_trading_analysis_{historical_period}.json'
    nfts_traits=pd.read_csv(nfts_traits_pth)
    collection_description=nfts_traits['description'][0]
    collection_historical_data=process_collection_trading_data_by_identifier(historical_collection_data_pth)
    
    with open(args.prompt, 'r') as f:   
        collection_prompt = f.read()
    # Initialize the tester
    tester = LLMTester()
    prompt=collection_prompt.replace("$[Historical Start]$", historical_start_date[historical_period]).replace("$[Historical End]$", historical_end).replace("$[Collection Description]$", collection_description).replace("$[Collection Historical Trading data]$", collection_historical_data)
    

    if args.model in ['gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18', 'gpt-4.1-2025-04-14', 'gpt-3.5-turbo-0125']:
        collection_result=tester.test_openai(prompt, model=args.model)
    elif args.model in ['claude-3-7-sonnet-20250219', 'claude-3-5-haiku-20241022']:
        collection_result=tester.test_anthropic(prompt, model=args.model)
    elif args.model in ['gemini-1.5-flash']:
        collection_result=tester.test_gemini(prompt, model=args.model)
    
    with open(output_file, 'w') as f:
        json.dump(collection_result, f, indent=2)
    
if __name__ == "__main__":
    main()