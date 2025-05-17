import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from tqdm import tqdm
import csv
import tarfile
import io

def load_text(text_name, lang):
    text_dir = os.path.join('text', lang, text_name+'.txt')
    with open(text_dir, 'r') as f:
        return f.readlines()
    
def load_model():
    config = XttsConfig()
    config.load_json("../xtts_ckpt/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="../xtts_ckpt/", use_deepspeed=True)
    model.cuda()
    return model

import random
def load_reference_audio(model, lang, proc_id):
    ref_dir = os.path.join('audio_prompts', lang)
    ref_files = os.listdir(ref_dir)
    ref_files = [file for file in ref_files if file.endswith('.wav')]

    random.shuffle(ref_files)
    ref_files = ref_files[:1600]

    gpt_cond_latent_lst = []
    speaker_embedding_lst = []
    print('\nLoading reference audio...')
    # for f in tqdm(ref_files[proc_id::8]):
    for f in tqdm(ref_files):
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[os.path.join(ref_dir, f)])
        gpt_cond_latent_lst.append(gpt_cond_latent)
        speaker_embedding_lst.append(speaker_embedding)
    return gpt_cond_latent_lst, speaker_embedding_lst


def load_reference_audio_names():
    ref_dir = 'audio_prompts/'
    ref_files = os.listdir(ref_dir)
    ref_files = [os.path.join(ref_dir, file) for file in ref_files if file.endswith('.wav')]
    random.shuffle(ref_files)
    return ref_files


def inference(model, sentence, lang, ref_file):
    # ref_dir = os.path.join('reference_audio', 'long')
    gpt_cond_latent, speaker_embedding = \
        model.get_conditioning_latents(
            audio_path=[ref_file],
            gpt_cond_len=16,
            gpt_cond_chunk_len=4,
            )
    out = model.inference(
        sentence,
        lang,
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
    )
    audio_wavs = out["wavs"]
    return audio_wavs

import click
@click.command()
@click.option("--text_name", type=str)
@click.option("--lang", type=str)
@click.option("--proc_id", type=int)
def main(text_name, lang, proc_id):
    print(text_name, lang)
    sentences = load_text(text_name, lang)
    print('\nNumber of sentences: ', len(sentences))

    save_dir = os.path.join(os.getenv('SYNDATA_PATH'), 'syndata', lang, text_name, f'proc_{proc_id}')
    os.makedirs(save_dir, exist_ok=True)

    tts_model = load_model()

    ref_files = load_reference_audio_names()
    
    sentences = sorted(sentences, key=len, reverse=True)
    
    sentences = sentences[proc_id::8]

    csv_path = os.path.join(os.getenv('SYNDATA_PATH'), 'syndata', lang, text_name, f'metadata_{proc_id}.csv')
    tar_path = os.path.join(os.getenv('SYNDATA_PATH'), 'syndata', lang, text_name, f'proc_{proc_id}.tar')
    with tarfile.open(tar_path, 'w') as tar, open(csv_path, 'w') as f:
        f.write('audio_path,transcript,language\n')
        writer = csv.writer(f, delimiter=',')
        
        batch_size = 20
        idx = -1
        for batch_start in tqdm(range(0, len(sentences), batch_size)):
            batch_end = min(batch_start + batch_size, len(sentences))
            batch_sentences = [s.strip() for s in sentences[batch_start:batch_end]]
            idx += 1
            ref_file = ref_files[idx%len(ref_files)]
            audio_wavs = inference(tts_model, batch_sentences, lang, ref_file)
            for batch_idx, (audio_wav, sentence) in enumerate(zip(audio_wavs, batch_sentences)):
                audio_wav = audio_wav.unsqueeze(0)
                audio_buffer = io.BytesIO()
                torchaudio.save(audio_buffer, audio_wav, 24000, format='wav')
                audio_buffer.seek(0)
                audio_fname = f'{idx}_{batch_idx}.wav'
                tarinfo = tarfile.TarInfo(name=audio_fname)
                tarinfo.size = audio_buffer.getbuffer().nbytes
                
                tar.addfile(tarinfo, audio_buffer)

                shorten_audio_path = os.path.join(f'proc_{proc_id}', audio_fname)
                writer.writerow([shorten_audio_path, sentence, lang])


if __name__ == "__main__":
    main()