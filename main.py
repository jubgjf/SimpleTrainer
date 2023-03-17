import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import SimpleTrainer as st


class Config(st.Config):
    def __init__(self):
        self.device = "cuda"
        self.checkpoint = "bert-base-uncased"
        self.wandb = "disabled"
        self.max_epochs = 2
        self.batch_size = 8
        self.lr = 5e-4


config = Config()
config.argparse()


class SSTModule(st.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(config.checkpoint, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.logger = st.ConsoleLogger()
        self.metrics = evaluate.load("accuracy")
        self.config = config

    def on_train_batch(self, batch, batch_idx) -> torch.Tensor:
        inputs = self.tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.config.device)
        inputs["labels"] = batch["label"].to(self.config.device)
        loss = self.model(**inputs).loss
        self.log("loss", loss.item())
        return loss

    def on_valid_batch(self, batch, batch_idx):
        inputs = self.tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.config.device)
        logits = self.model(**inputs).logits
        preds = torch.argmax(logits, dim=-1)
        self.metrics.add_batch(predictions=preds, references=batch["label"])

    def after_valid_batch(self):
        results = self.metrics.compute()
        self.log_dict(results)

    def on_test_batch(self, batch, batch_idx):
        self.on_valid_batch(batch, batch_idx)

    def after_test_batch(self):
        self.after_valid_batch()


if __name__ == "__main__":
    st.seed_everything(42)

    train_dataset = load_dataset("sst2", split="train[:20%]")
    valid_dataset = load_dataset("sst2", split="validation[:20%]")
    test_dataset = load_dataset("sst2", split="test[:20%]")

    trainer = st.Trainer(
        module=SSTModule(),
        max_epochs=config.max_epochs,
        device=config.device,
    )

    trainer.train(
        DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False),
    )
    trainer.test(
        DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
    )
