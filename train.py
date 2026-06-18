import csv, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

model_name = 'Qwen/Qwen3-1.7B'

tokenizer = AutoTokenizer.from_pretrained(model_name)

data = []
with open('/mnt/8TB_HDD/datasets/JSP/train.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append({
            'prompt': [{'role': 'user', 'content': f'EN2JA: {row[1]}'}],
            'completion': [{'role': 'assistant', 'content': f'JA: {row[2]}'}],
        })

dataset = Dataset.from_list(data)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16
)

sft_config = SFTConfig(
    output_dir='./qwen_model',
    bf16=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    completion_only_loss=True,
    num_train_epochs=1,
    logging_steps=50,
    dataloader_num_workers=8,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=4,
    dataset_num_proc=24,
    max_length=128,
    packing=True,
    gradient_checkpointing=True,
    save_strategy='steps',
    save_steps=200,
    save_total_limit=3,
    assistant_only_loss=True,
    eos_token=tokenizer.eos_token,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer
)

trainer.train(resume_from_checkpoint=True)

# accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 train.py
