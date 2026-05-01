import yaml
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    # Load model and tokenizer via Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Load and prepare data
    df = pd.read_csv(config["train_data_path"])
    dataset = Dataset.from_pandas(df)

    EOS_TOKEN = tokenizer.eos_token

    def formatting_func(example):
        if isinstance(example.get("text"), list):
            output_texts = []
            for text, intent in zip(example["text"], example["intent"]):
                prompt = f"Below is a customer message to a bank. Classify the intent of the message into the correct category.\n\n### Message:\n{text}\n\n### Intent:\n{intent}{EOS_TOKEN}"
                output_texts.append(prompt)
            return output_texts
        else:
            text = example["text"]
            intent = example["intent"]
            return f"Below is a customer message to a bank. Classify the intent of the message into the correct category.\n\n### Message:\n{text}\n\n### Intent:\n{intent}{EOS_TOKEN}"

    # Configure SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            warmup_steps=5,
            num_train_epochs=config["num_epochs"],
            learning_rate=config["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim=config["optimizer"],
            weight_decay=config["weight_decay"],
            lr_scheduler_type=config["lr_scheduler_type"],
            seed=3407,
            output_dir=config["output_dir"],
        ),
    )

    # Start training
    trainer_stats = trainer.train()

    # Save the model
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
