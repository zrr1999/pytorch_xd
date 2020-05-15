import torch
import tqdm


class TrainerBase(object):

    def __init__(self, device="cpu"):
        """

        Args:
            device: 训练使用的硬件设备
        """
        self.model = None
        self.optimizer = None
        self.dataloader = None

        self.device = device
        print(f"GPU 可用状态：{torch.cuda.is_available()}，使用状态：{self.device != 'cpu'}")

    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)

    def train(self, model, max_epoch, model_name="UntitledModel"):
        self.model = model.to(self.device)
        self.dataloader = self.train_dataloader()
        self.optimizer = self.configure_optimizers(model)
        for epoch in range(1, max_epoch + 1):
            sum_loss = 0
            training_results = []
            t_dataloader = tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                                     desc=f"正在训练 Epoch {epoch}", postfix={"loss": None})
            for batch_idx, batch in t_dataloader:
                inp, label = batch
                batch = inp.to(self.device), label.to(self.device)

                training_result = self.training_step(batch, batch_idx)
                training_results.append(training_result)

                loss = training_result["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss.detach_()

                sum_loss += loss
                t_dataloader.set_postfix(loss=training_result["loss"].item())
            avg_loss = sum_loss / len(self.dataloader)
            print("\n{}(Epoch {}) 平均误差 = {}".format(model_name, epoch, avg_loss))
            self.save(f"./pretrained/{model_name}_epoch{epoch}_end.pth")
            self.training_epoch_end(training_results)
        self.validate(self.model, model_name)

    def validate(self, model, model_name):
        self.model = model.to(self.device)
        self.dataloader = self.val_dataloader()
        self.optimizer = self.configure_optimizers(model)

        # self.dataset.set_mode("val", self.n)
        validation_results = []
        t_dataloader = tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                                 desc=f"正在验证", postfix={"loss": None})
        for batch_idx, batch in t_dataloader:
            inp, label = batch
            batch = inp.to(self.device), label.to(self.device)

            validation_result = self.validation_step(batch, batch_idx)
            validation_results.append(validation_result)

            loss = validation_result["val_loss"]
            loss.detach_()

            t_dataloader.set_postfix(loss=loss.item())
        avg_loss = torch.stack([r["val_loss"] for r in validation_results]).mean()
        accuracy = torch.stack([r["val_accuracy"] for r in validation_results]).mean()
        # print('Epoch:{} iter{} loss:{}'.format(epoch, ite, loss.item()))
        print("\n{} 平均误差 = {} 正确率 = {}".format(model_name, avg_loss, accuracy))
        self.validation_epoch_end(validation_results)

    def test(self, model):
        self.model = model.to(self.device)
        self.dataloader = self.test_dataloader()
        self.optimizer = self.configure_optimizers(model)

        # self.dataset.set_mode("val", self.n)
        test_results = []
        t_dataloader = tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                                 desc=f"正在验证", postfix={"loss": None})
        for batch_idx, batch in t_dataloader:
            inp, label = batch
            batch = inp.to(self.device), label.to(self.device)

            test_result = self.test_step(batch, batch_idx)
            test_results.append(test_result)

            loss = test_result["val_loss"]
            loss.detach_()

            t_dataloader.set_postfix(loss=test_result["loss"].item())
        avg_loss = torch.stack([r["loss"] for r in test_results]).mean()
        accuracy = torch.stack([r["accuracy"] for r in test_results]).mean()
        # print('Epoch:{} iter{} loss:{}'.format(epoch, ite, loss.item()))
        print("模型: {}  平均误差 = {} 正确率 = {}".format("暂未实现", avg_loss, accuracy))
        self.validation_epoch_end(test_results)

    def save(self, path):
        torch.save(self.model, path)

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def configure_optimizers(self, model):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
