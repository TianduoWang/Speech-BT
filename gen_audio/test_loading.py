import os
from datasets import load_dataset
import click
@click.command()
@click.option('--lang', type=str, help='Language code')
@click.option('--text_name', type=str, help='Text name')
def main(lang, text_name):
    print(f"Loading dataset for {lang} with text name {text_name}")
    dataset = load_dataset(
        f"{os.getenv('SYNDATA_PATH')}/syndata/{lang}/{text_name}", 
        trust_remote_code=True, 
        num_proc=8,
        )
    print(dataset)

if __name__ == "__main__":
    main()