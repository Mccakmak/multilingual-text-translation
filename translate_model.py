import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm



def load_model():
    model_name = 'Helsinki-NLP/opus-mt-mul-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def translate_text(text, tokenizer, model, device, progress_bar):
    if not isinstance(text, str) or pd.isna(text):
        progress_bar.update(1)  # Update progress bar even if text is not valid
        return text  # Return the original text as it is
    try:
        #if detect(text) != 'en':
        batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(**batch, max_new_tokens=100)
        translation = tokenizer.batch_decode(gen, skip_special_tokens=True)
        return translation[0]
        #else:
        #    return text
    except Exception as e:
        print(f"Error during translation: {e}")
        print(f"Problematic text: {text}")
    finally:
        progress_bar.update(1)
    return text



def translate_column(data, column, tokenizer, model, device):
    return [translate_text(text, tokenizer, model, device) for text in tqdm(data[column], desc=f"Translating {column}")]

def read_csv(file_path, columns):
    df = pd.read_csv(file_path)
    return df, df[columns].copy()

def save_to_csv(df, original_columns, translated_columns, file_path):
    for col in original_columns:
        df[col] = translated_columns[col]
    new_file_path = os.path.splitext(file_path)[0] + "_translated.csv"
    df.to_csv(new_file_path, index=False)

def translate(file_path, columns_to_translate):
    tokenizer, model, device = load_model()
    df, columns_data = read_csv(file_path, columns_to_translate)

    for col in columns_to_translate:
        with tqdm(total=len(columns_data[col]), desc=f"Translating {col}") as pbar:
            with ThreadPoolExecutor() as executor:
                # Create a lambda function that includes the progress bar
                translate_func = lambda x: translate_text(x, tokenizer, model, device, pbar)
                translated_texts = list(executor.map(translate_func, columns_data[col]))
                columns_data[col] = translated_texts

    save_to_csv(df, columns_to_translate, columns_data, file_path)


if __name__ == '__main__':
    file_path = 'SCS/asonam_scs_video_depth.csv'
    columns_to_translate = ['title', 'description']
    translate(file_path, columns_to_translate)