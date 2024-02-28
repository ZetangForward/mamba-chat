import torch
import argparse
import os
import sys
sys.path.append("/nvme/zecheng/modelzipper/projects/state-space-model")
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from trainer.data import ChatDataModule
from trainer.mamba_trainer import MambaTrainer
from custom_dataset.data import TextFillingDataset
from modelzipper.tutils import *


def run(args):
        
    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    
    content = auto_read_data("/nvme/zecheng/data/roc_stories/processed_2017/train.jsonl")
    kwargs = {"max_text_length": 128}
    train_dataset = TextFillingDataset(content[1000:], tokenizer, "train", full_modeling=True, **kwargs)
    valid_dataset = TextFillingDataset(content[:1000], tokenizer, "valid", full_modeling=True, **kwargs)

    data_module = dict(
        train_dataset=train_dataset, 
        eval_dataset=valid_dataset, 
        data_collator=None
    )

    trainer = MambaTrainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output_dir,
            logging_steps=50,
            save_steps=500,
            report_to="tensorboard",
            remove_unused_columns=False
        ),
        **data_module,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/nvme/zecheng/ckpt/mamba-chat")
    parser.add_argument("--model", type=str, default="state-spaces/mamba-2.8b")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    run(args)
