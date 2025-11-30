"""
Test script for all dataset-specific table extractors
"""
import json
from finqa_table_extraction import FinQATableExtractor
from tatqa_table_extraction import TATQATableExtractor
from multihiertt_table_extraction import MultiHierttTableExtractor
from convfinqa_table_extraction import ConvFinQATableExtractor


def test_dataset(dataset_name, extractor_class, corpus_path):
    """Test a specific dataset extractor"""
    print("=" * 80)
    print(f"Testing {dataset_name}")
    print("=" * 80)
    
    try:
        with open(corpus_path, 'r') as f:
            doc = json.loads(f.readline())
        
        extractor = extractor_class()
        
        print(f"\ncorpus_id: {doc['_id']}")
        
        result = extractor.extract_table(doc['text'])
        if result:
            text_before, table = result
            print("\nText before table:")
            print(text_before[:300] if len(text_before) > 300 else text_before)
            
            print("\nRaw Table (first 500 chars):")
            print(table[:500])
            
            print("\nParsed Table:")
            parsed = extractor.parse_table_to_text(table)
            print(parsed)
        else:
            print("No table found")
    
    except FileNotFoundError:
        print(f"File not found: {corpus_path}")
        print("Please update the path in this script")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n")


def main():
    """Test all datasets"""
    
    # Update these paths to match your file locations
    datasets = [
        ("FinQA", FinQATableExtractor, './dataset/FinQA/corpus.jsonl'),
        ("TATQA", TATQATableExtractor, './dataset/TATQA/corpus.jsonl'),
        ("MultiHiertt", MultiHierttTableExtractor, './dataset/MultiHiertt/corpus.jsonl'),
        ("ConvFinQA", ConvFinQATableExtractor, './dataset/ConvFinQA/corpus.jsonl'),
    ]
    
    for dataset_name, extractor_class, corpus_path in datasets:
        test_dataset(dataset_name, extractor_class, corpus_path)
        print("\n")


if __name__ == "__main__":
    main()