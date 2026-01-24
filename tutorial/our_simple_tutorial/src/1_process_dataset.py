"""
Dataset Processing Script
Processes downloaded datasets into unified format
"""

import pandas as pd
import os
from pathlib import Path
from datasets import load_from_disk
from dataset_processor import ClinicalDialogueProcessor


def process_single_turn():
    """Process single-turn QA dataset"""
    print("\nProcessing single-turn QA dataset: medical_meadow_medical_flashcards")
    
    try:
        # Load data
        data_dir = Path("../data/raw/medical_meadow_medical_flashcards")
        if not data_dir.exists():
            print(f"Data not found: {data_dir}")
            print("Please run 0_download_dataset.py first")
            return None
            
        dataset = load_from_disk(str(data_dir))
        print(f"Loaded dataset: {len(dataset)} samples")
        
        # Create processor
        processor = ClinicalDialogueProcessor(data_dir="../data")
        
        # Process data
        processed_data = processor.process_single_turn_qa(
            dataset,
            question_col="input",
            answer_col="output"
        )
        
        print(f"Processed: {len(processed_data)} samples")
        
        # Split dataset
        split_data = processor.split_dataset(processed_data)
        
        # Save to data/processed/medical_meadow/
        dataset_dir = processor.output_dir / "medical_meadow"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        processor.output_dir = dataset_dir
        
        processor.save_to_json(split_data["train"], "train.json")
        processor.save_to_json(split_data["validation"], "val.json")
        processor.save_to_json(split_data["test"], "test.json")
        
        # Statistics
        stats = processor.get_statistics(processed_data)
        processor.print_statistics(stats)
        
        # Preview samples
        processor.preview_samples(processed_data, n=2)
        
        return processed_data
        
    except Exception as e:
        print(f"Failed to process single-turn dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_multi_turn():
    """Process multi-turn dialogue dataset"""
    print("\nProcessing multi-turn dataset: ChatDoctor-HealthCareMagic")
    
    try:
        # Try to load main dataset
        data_dir = Path("../data/raw/ChatDoctor-HealthCareMagic")
        if not data_dir.exists():
            # Try backup dataset
            data_dir = Path("../data/raw/ai-medical-chatbot")
            if not data_dir.exists():
                print("Data not found. Please run 0_download_dataset.py first")
                return None
        
        dataset = load_from_disk(str(data_dir))
        print(f"Loaded dataset: {len(dataset)} samples")
        print(f"Columns: {dataset.column_names}")
        
        # Create processor
        processor = ClinicalDialogueProcessor(data_dir="../data")
        
        # Check data structure
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        
        # Process data
        processed_data = []
        
        for item in dataset:
            try:
                messages = []
                
                # Format 1: instruction + input + output 
                if "instruction" in item and "input" in item and "output" in item:
                    if item.get("instruction"):
                        messages.append({"role": "system", "content": str(item["instruction"])})
                    
                    input_text = str(item.get("input", ""))
                    output_text = str(item.get("output", ""))
                    
                    if "\nDoctor:" in input_text or "\nPatient:" in input_text:
                        # Parse multi-turn dialogue
                        turns = []
                        combined_text = input_text + "\n" + output_text
                        
                        for line in combined_text.split("\n"):
                            if line.startswith("Patient:") or line.startswith("patient:"):
                                turns.append({"role": "user", "content": line.split(":", 1)[-1].strip()})
                            elif line.startswith("Doctor:") or line.startswith("doctor:"):
                                turns.append({"role": "assistant", "content": line.split(":", 1)[-1].strip()})
                        
                        if turns:
                            messages.extend(turns)
                    else:
                        # Single turn
                        if input_text:
                            messages.append({"role": "user", "content": input_text})
                        if output_text:
                            messages.append({"role": "assistant", "content": output_text})
                
                # Format 2: Patient + Doctor columns
                elif "Patient" in item and "Doctor" in item:
                    patient_text = str(item.get("Patient", ""))
                    doctor_text = str(item.get("Doctor", ""))
                    
                    if patient_text:
                        messages.append({"role": "user", "content": patient_text})
                    if doctor_text:
                        messages.append({"role": "assistant", "content": doctor_text})
                
                # Format 3: messages column
                elif "messages" in item:
                    messages = item["messages"]
                
                # Format 4: conversation column
                elif "conversation" in item:
                    conv = item["conversation"]
                    if isinstance(conv, str):
                        import json
                        try:
                            conv = json.loads(conv)
                        except:
                            pass
                    if isinstance(conv, list):
                        messages = conv
                
                # Add valid samples
                if messages and len(messages) >= 2:
                    processed_data.append({
                        "messages": messages,
                        "type": "multi_turn"
                    })
                    
            except Exception as e:
                continue
        
        print(f"Processed: {len(processed_data)} samples")
        
        if len(processed_data) == 0:
            print("Warning: No valid multi-turn data processed")
            return None
        
        # Split dataset
        split_data = processor.split_dataset(processed_data)
        
        # Save to data/processed/chatdoctor/
        dataset_name = "chatdoctor" if "ChatDoctor" in str(data_dir) else "medical_chatbot"
        dataset_dir = processor.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        processor.output_dir = dataset_dir
        
        processor.save_to_json(split_data["train"], "train.json")
        processor.save_to_json(split_data["validation"], "val.json")
        processor.save_to_json(split_data["test"], "test.json")
        
        # Statistics
        stats = processor.get_statistics(processed_data)
        processor.print_statistics(stats)
        
        # Preview samples
        processor.preview_samples(processed_data, n=2)
        
        return processed_data
        
    except Exception as e:
        print(f"Failed to process multi-turn dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_datasets():
    """Merge single-turn and multi-turn datasets (optional)"""
    print("\nMerging datasets")
    
    try:
        processor = ClinicalDialogueProcessor(data_dir="../data")
        
        # Load single-turn and multi-turn data
        single_turn_path = Path("../data/processed/medical_meadow/train.json")
        multi_turn_path_1 = Path("../data/processed/chatdoctor/train.json")
        multi_turn_path_2 = Path("../data/processed/medical_chatbot/train.json")
        
        if not single_turn_path.exists():
            print(f"Single-turn data not found: {single_turn_path}")
            return None
        
        processor.output_dir = Path("../data/processed/medical_meadow")
        single_turn_train = processor.load_from_json("train.json")
        
        # Try to load multi-turn data
        multi_turn_train = []
        if multi_turn_path_1.exists():
            processor.output_dir = Path("../data/processed/chatdoctor")
            multi_turn_train = processor.load_from_json("train.json")
        elif multi_turn_path_2.exists():
            processor.output_dir = Path("../data/processed/medical_chatbot")
            multi_turn_train = processor.load_from_json("train.json")
        
        # Convert to unified chat format
        single_turn_chat = processor.convert_to_chat_format(single_turn_train)
        multi_turn_chat = processor.convert_to_chat_format(multi_turn_train)
        
        # Merge
        merged_train = single_turn_chat + multi_turn_chat
        
        # Save to data/processed/merged/
        processor.output_dir = Path("../data/processed/merged")
        processor.output_dir.mkdir(parents=True, exist_ok=True)
        processor.save_to_json(merged_train, "train.json")
        
        print(f"Merge complete:")
        print(f"  Single-turn: {len(single_turn_chat)} samples")
        print(f"  Multi-turn: {len(multi_turn_chat)} samples")
        print(f"  Total: {len(merged_train)} samples")
        
        return merged_train
        
    except Exception as e:
        print(f"Failed to merge datasets: {e}")
        return None

def process_data_for_verl(raw_data_path, output_dir):
    """Convert raw data to verl-compatible Parquet format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data (Mocking clinical data here)
    raw_data = [
        {
            "instruction": "User: I have a headache and nausea.\nDoctor:",
            "label": "I suggest you visit the neurology department immediately..."
        }
        # ... load actual data here
    ]

    df = pd.DataFrame(raw_data)
    
    # 1. SFT Format: Requires 'prompt' and 'response'
    sft_df = pd.DataFrame()
    sft_df['prompt'] = df['instruction']
    sft_df['response'] = df['label']
    
    # 2. GRPO/RL Format
    # 'reward_model_input': used for reward calculation (usually prompt)
    # 'ground_truth': used for rule-based reward scoring
    rl_df = pd.DataFrame()
    rl_df['data_source'] = ['clinical_medical'] * len(df)
    rl_df['prompt'] = [{
        "role": "user", "content": x
    } for x in df['instruction']] # Chat format preferred
    rl_df['ability'] = ["medical"] * len(df)
    rl_df['reward_model_input'] = rl_df['prompt'] 
    rl_df['ground_truth'] = df['label']

    # Save to Parquet
    sft_df.to_parquet(os.path.join(output_dir, 'train_sft.parquet'))
    rl_df.to_parquet(os.path.join(output_dir, 'train_rl.parquet'))
    print(f"Data saved to {output_dir}")

def main():
    """Main function"""
    print("\nMedical Dialogue Dataset Processing Tool")
    print("-" * 50)
    
    # Process single-turn QA dataset
    single_turn_data = process_single_turn()
    
    # Process multi-turn dialogue dataset
    multi_turn_data = process_multi_turn()
    
    # Merge datasets
    if single_turn_data and multi_turn_data:
        merge_datasets()
    
    print("\nDataset processing complete!")
    print("Processed data saved in ../output/ directory")
    print("Next: Run 2_inference.py to test model inference")


if __name__ == "__main__":
    main()
