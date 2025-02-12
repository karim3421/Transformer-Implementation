import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from data import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weight_file_path, latest_weight_file

from datasets import load_dataset

import warnings

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from pathlib import Path

print("All imports successful")

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
    pass

def get_or_build_toenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
        print("Tokenizer saved to: ", tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_trg']}", split='train')
    print(f"Dataset size: {len(ds_raw)}")

    #building tokenizers
    tokenizer_src = get_or_build_toenizer(config, ds_raw, config['lang_src'])
    tokenizer_trg = get_or_build_toenizer(config, ds_raw, config['lang_trg'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size   = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])
    # print(f"Encoder Mask: {train_ds[0]['encoder_mask'].shape}")
    # print(f"Decoder Mask: {train_ds[0]['decoder_mask'].shape}")

    max_src_len = 0
    max_trg_len = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        trg_ids = tokenizer_trg.encode(item['translation'][config['lang_trg']]).ids

        max_src_len = max(max_src_len, len(src_ids))
        max_trg_len = max(max_trg_len, len(trg_ids))

    print(f"Max source length: {max_src_len}")
    print(f"Max target length: {max_trg_len}")

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg

# Get model
def get_model(config, vocab_src_len, vocab_trg_len):
    model = build_transformer(vocab_src_len, vocab_trg_len, config['seq_len'], config['seq_len'], config['d_model'] )
    return model

# Train model
def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents= True, exist_ok= True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)

    intial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_file_path = latest_weight_file(config) if preload == "latest" else get_weight_file_path(config, preload) if preload else None
    if model_file_path:
        print(f'Prelaoding model: {model_file_path}')
        state = torch.load(model_file_path)
        intial_epoch = state['epoch'] + 1
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(intial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc= f"Processing epoch: {epoch}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask  = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input , decoder_mask)
            project_output = model.linear(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(project_output.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        model_path = get_weight_file_path(config, f"{epoch}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_path)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

# config = get_config()
# train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)
# model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size())
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))




