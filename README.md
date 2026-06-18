# Japanese to English Translator
This project fine tunes Qwen3-1.7B as an English-to-Japanese translator using supervised fine-tuning. Training uses prompt-and-completion examples.
```
> python is a high level programming language, used for tasks like AI and websites
JA: pythonは 人工知能やウェブサイトなどの 高度なプログラミング言語です
> modern AI LLMs are used for language translation 
JA: 近代のAI LLMは翻訳に使われます
> 
```

To train the model use:
```py
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 train.py
```
For inference
```py
python main.py
```
