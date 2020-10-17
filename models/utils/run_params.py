class RunParams():
    def __init__(self, 
                model_name='gpt2',
                max_epochs=1,
                debug=True,
                batch_size=8,
                data_set_percentage=1,
                seq_length=32,
                statement_length=16,
                momentum=.9,
                lr=1e-2,
                repo='wikitext-103-raw-v1',
                num_workers=2,
                kb_statements_file=None,
                run_name='test',
                project_name='test-project'
            ):
            
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.debug = debug
        self.batch_size = batch_size
        self.data_set_percentage = data_set_percentage
        self.seq_length = seq_length
        self.statement_length = statement_length
        self.momentum = momentum
        self.lr = lr
        self.repo = repo
        self.num_workers = num_workers
        self.kb_statements_file = kb_statements_file

