class Client:
    def __init__(
        self,
        train_global,
        test_global,
        args,
        device,
        model_trainer,
    ):
        self.train_global = train_global
        self.test_global = test_global
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.train_global, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.test_global
        else:
            test_data = self.train_global
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
