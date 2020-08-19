from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import pytorch_lightning as pl
import torch
import sh
import nlp
import wandb
from pytorch_lightning.loggers import WandbLogger

# !wandb login e279feeab3d602ab530e4eb23df8ac3ff3763461

# Variables
model_name = 'gpt2'
epochs = 10
debug = True
batch_size = 8
percent = 100
dataset = 'wikitext'
seq_length = 32
momentum = .9
lr = 1e-2
repo = 'wikitext-103-raw-v1'

# The model to train on the scientific papers dataset
class WikitextLM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        config = GPT2Config()
        self.model = GPT2LMHeadModel(config)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def get_id(self, token):
        """Request a word from ConceptNet"""
        assert isinstance(token, str), "Request token not a string"
        endpoint = "http://wikidata.org/w/api.php?"
        action = "action=wbsearchentities&"
        search = "search=" + token + "&"
        language = "language=en&"
        format = "format=json"
        request = endpoint + action + search + language + format
        resource = requests.get(request)
        assert resource.status_code == requests.codes.ok
        try:
            id = resource.json()['search'][0]['id']
        except:
            id = None
        return id

    # Download and prepare data
    def prepare_data(self):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token 
        rake = Rake()
        client = Client()

        def _extract_knowledge(x):
            try:
                rake.extract_keywords_from_text(x['text'])
                ranked_phrases = rake.get_ranked_phrases()
                for phrase in ranked_phrases:
                    try:
                        id = self.get_id(phrase)
                        entity = client.get(id)
                        description = entity.attributes['descriptions']['en']['value']
                        break
                    except:
                        continue        
                label = entity.attributes['labels']['en']['value']
                statement = label + ":" + description
                x['statement'] = statement
            except:
                x['statement'] = ""
            return x
              

        def _tokenize(x):
            tokens = tokenizer(
                x['text'],
                max_length=seq_length,
                truncation=True,
                padding=True)
            x['input_ids'] = tokens['input_ids']
            x['attention_mask'] = tokens['attention_mask']
            tokens = tokenizer(
                x['statement'],
                max_length=statement_length,
                truncation=True,
                padding=True)
            x['statement_ids'] = tokens['input_ids']
            x['statement_mask'] = tokens['attention_mask']
            return x

        def _prepare_ds(split):
            ds = nlp.load_dataset(dataset, repo, split=f'{split}[:{batch_size if debug else f"{percent}%"}]')
            ds = ds.map(_extract_knowledge)
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'statement_ids', 'statement_mask'])
            return ds

        self.train_ds, self.val_ds = map(_prepare_ds, ('train', 'validation'))

    def forward(self, inputs):
        # Only call if training
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        loss = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)[0]
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
                batch_size=batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                ) 

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                )


    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=lr,
            momentum=momentum,
        ) 

model = WikitextLM()
trainer = pl.Trainer(
    default_root_dir='logs',
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=epochs,
    fast_dev_run=debug,
    logger=WandbLogger(save_dir='logs/', name='wikitext-no-kb', project='experiment-1'),
)

trainer.fit(model)
trainer.save_checkpoint("model.ckpt")