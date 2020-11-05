from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained(self.run_params['model_name'])
tokenizer.pad_token = tokenizer.eos_token 

# Load, Tokenize, and Augment data
def prepare_data(run_params):
    train_ds, val_ds = map(prepare_ds, ('train', 'validation'))
    return train_ds, val_ds

# Tokenize a sequence
def tokenize(x):
    tokens = tokenizer(
        x['statement'],
        max_length=run_params['statement_length'],
        padding=True)
    x['statement_ids'] = tokens['input_ids']
    x['statement_mask'] = tokens['attention_mask']
    return x

# Tokenize a sequence
def tokenize(x):
    tokens = tokenizer(
        x['text'],
        max_length=run_params.seq_length,
        padding=True)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    return input_ids, attention_mask

# Load the data
def prepare_ds(split):
    ds = load_dataset('text', data_files=run_params.data_files)
    wikidata_client = WikidataClient(run_params['data_file'])
    ds = ds.map(wikidata_client.extract_knowledge)
    wikidata_client.close()
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'statement_ids', 'statement_mask'])

    return ds
