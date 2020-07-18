from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pytorch_lightning as pl
import torch
import sh
import nlp
from absl import app, flags, logging

# Delete and create logs folder each time you train
sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

# Flags for parameters that can be specified from command line
flags.DEFINE_string('model', 'gpt2', '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_boolean('debug', True, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_integer('percent', 5, '')
flags.DEFINE_string('dataset', 'squad', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_float('lr', 1e-2, '')

FLAGS = flags.FLAGS

# The model to train on SQuAd
class SQuADLM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(FLAGS.model)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    # Download and prepare data
    def prepare_data(self):
        tokenizer = GPT2Tokenizer.from_pretrained(FLAGS.model)
        tokenizer.pad_token = tokenizer.eos_token 
        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(
                x['question'],
                max_length=FLAGS.seq_length,
                pad_to_max_length=True)['input_ids']
            answers = [a['text'][0] for a in x['answers']]
            x['label'] = tokenizer.batch_encode_plus(
                answers,
                max_length=FLAGS.seq_length,
                pad_to_max_length=True)['input_ids']
            return x

        def _prepare_ds(split):
            ds = nlp.load_dataset(FLAGS.dataset, split=f'{split}[:{FLAGS.batch_size if FLAGS.debug else f"{FLAGS.percent}%"}]')
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.val_ds = map(_prepare_ds, ('train', 'validation'))

    def forward(self, input_ids, label):
        mask = (input_ids != 50256).float()
        loss, prediction_scores, _ = self.model(input_ids, labels=label, attention_mask=mask)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch['input_ids'], batch['label'])
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch['input_ids'], batch['label'])
        return {'loss': loss, 'log': {'val_loss': loss}}
    
    def validation_epoch_end(self, outputs):
        import IPython ; IPython.embed() ; exit(1)
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        import IPython ; IPython.embed() ; exit(1)
        out = {'val_loss': loss}
        import IPython ; IPython.embed() ; exit(1)
        return {**outputs, 'log': outputs}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                self.train_ds,
                batch_size=FLAGS.batch_size,
                drop_last=True,
                shuffle=True,
                ) 

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=FLAGS.batch_size,
                drop_last=True,
                shuffle=True,
                ) 

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
        ) 

def main(_):
    model = SQuADLM()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='squad', version=0),
    )
    trainer.fit(model)

if __name__ == '__main__':
    app.run(main)