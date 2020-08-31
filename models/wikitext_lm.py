from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import pytorch_lightning as pl
import torch
import nlp
from models.utils.wikidata_client import WikidataClient

# The model to train on the scientific papers dataset
class WikitextLM(pl.LightningModule):
    """
        Language model trained on the wikitext dataset
    """
    def __init__(self, run_params):
        super().__init__()
        self.run_params = run_params
        config = GPT2Config()
        self.model = GPT2LMHeadModel(config)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        

    # Download and prepare data
    def prepare_data(self):
        tokenizer = GPT2Tokenizer.from_pretrained(self.run_params['model_name'])
        tokenizer.pad_token = tokenizer.eos_token 
        self.EOS = tokenizer.pad_token
            
        def _tokenize(x):
            tokens = tokenizer(
                x['text'],
                max_length=self.run_params['seq_length'],
                truncation=True,
                padding=True)
            x['input_ids'] = tokens['input_ids']
            x['attention_mask'] = tokens['attention_mask']
            tokens = tokenizer(
                x['statement'],
                max_length=self.run_params['statement_length'],
                truncation=True,
                padding=True)
            x['statement_ids'] = tokens['input_ids']
            x['statement_mask'] = tokens['attention_mask']
            return x

        def _prepare_ds(split):
            dataset = self.run_params['dataset']
            repo = self.run_params['repo']
            batch_size = self.run_params['batch_size']
            debug = self.run_params['debug']
            self.run_params['percent']

            ds = nlp.load_dataset(dataset, repo, split=f'{split}[:{batch_size if debug else f"{percent}%"}]')
            wikidata_client = WikidataClient()
            ds = ds.map(wikidata_client.extract_knowledge)
            wikidata_client.data_writer.close()
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'statement_ids', 'statement_mask'])
            return ds

        self.train_ds, self.val_ds = map(_prepare_ds, ('train', 'validation'))

    def forward(self, inputs):
        x = torch.cat((inputs['input_ids'], inputs['statement_ids']), -1)   
        attention_mask = torch.cat((inputs['attention_mask'], inputs['statement_mask']), -1)  
        padding = torch.full_like(inputs['statement_ids'], fill_value=50256, dtype=torch.long, device=torch.device('cuda'))   
        labels = torch.cat((inputs['input_ids'], padding), dim=-1)
        loss = self.model(x, attention_mask=attention_mask, labels=labels)[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return {'loss': loss, 'log': {'val_loss': loss}}
    
    def validation_epoch_end(self, outputs):
        loss = torch.cat([o['loss'].unsqueeze(0) for o in outputs], 0).mean()
        out = {'val_loss': loss}
        return {**out, 'log': out}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                self.train_ds,
                batch_size=self.run_params['batch_size'],
                drop_last=True,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                ) 

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=self.run_params['batch_size'],
                drop_last=True,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                )


    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.run_params['lr'],
            momentum=self.run_params['momentum'],
        )  
