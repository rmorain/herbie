from datasets import load_dataset

# Load, Tokenize, and Augment data
def prepare_data(run_params):
    def _tokenize(x):
        tokens = tokenizer(
            x['text'],
            max_length=run_params['seq_length'],
            truncation=True,
            padding=True)
        x['input_ids'] = tokens['input_ids']
        x['attention_mask'] = tokens['attention_mask']
        tokens = tokenizer(
            x['statement'],
            max_length=run_params['statement_length'],
            truncation=True,
            padding=True)
        x['statement_ids'] = tokens['input_ids']
        x['statement_mask'] = tokens['attention_mask']
        return x

    def _prepare_ds(split):
        dataset = run_params['dataset']
        repo = run_params['repo']
        batch_size = run_params['batch_size']
        debug = run_params['debug']
        percent = run_params['percent']

        ds = load_dataset(dataset, repo, split=f'{split}[:{batch_size if debug else f"{percent}%"}]')
        wikidata_client = WikidataClient(run_params['data_file'])
        ds = ds.map(wikidata_client.extract_knowledge)
        wikidata_client.close()
        ds = ds.map(_tokenize, batched=True)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'statement_ids', 'statement_mask'])

        return ds
    train_ds, val_ds = map(_prepare_ds, ('train', 'validation'))
    return train_ds, val_ds

# Load the dataset
def load_data(run_params):
    dataset = load_dataset('text', data_files=run_params.data_files)
