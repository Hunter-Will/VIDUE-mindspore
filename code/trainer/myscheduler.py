class MyScheduler:
    """
    This is just a placeholder for the code to run, do not have any other effects. Don't be fool.
    """
    def __init__(self, optimizer, mode='min', factor=0.5, patience=1, verbose=True):
        self.optimizer=optimizer
        self.mode=mode
        self.factor=factor
        self.patience=patience
        self.verbose=verbose
        self.last_epoch=0
        
    def step(self,loss):
        '''
        TODO: use loss calc lr
        '''
        pass
        # ms do not support change lr during training, so maybe wait for their developers finish this.
        self.last_epoch+=1