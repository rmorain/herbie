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

FLAGS = flags.FLAGS

# The model to train on SQuAd
class SQuADLM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(FLAGS.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        # self.model = GPT2LMHeadModel.from_pretrained(FLAGS.model, pad_token_id=self.tokenizer.eos_token_id)
        # self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    # Download and prepare data
    def prepare_data(self):
        def _tokenize(x):
            x['input_ids'] = self.tokenizer.batch_encode_plus(
                x['question'],
                max_length=FLAGS.seq_length,
                pad_to_max_length=True)['input_ids']
            # answers = [a['text'][0] for a in x['answers']]
            # x['label'] = self.tokenizer.batch_encode_plus(
            #     answers,
            #     max_length=FLAGS.seq_length,
            #     pad_to_max_length=True)['input_ids']
            # print('here')
            return x
        
        def _prepare_ds(split):
            ds = nlp.load_dataset(FLAGS.dataset, split=f'{split}[:{FLAGS.batch_size if FLAGS.debug else f"{FLAGS.percent}%"}]')
            ds = ds.map(_tokenize, batched=True)
            import IPython ; IPython.embed() ; exit(1)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            import IPython ; IPython.embed() ; exit(1)
            return ds

        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))

    def forward(self, input_ids):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def train_dataloader(self):
        pass 

    def val_dataloader(self):
        pass 

    def configure_optimizers(self):
        pass 

def main(_):
    model = SQuADLM()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0),
    )
    trainer.fit(model)

if __name__ == '__main__':
    app.run(main)