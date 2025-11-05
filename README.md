# Japaense to English Translator
The model uses a transformer encoder to learn the patterns of the English text, which is then fed into a transformer decoder that predicts each token of the Japanese text. Separate language and positional embeddings are used for Japanese and English. Adaptive softmax is used to reduce VRAM and compute.
```
> cat
猫
> cat girls
猫の子
> 
```

To train the model use:
```py
python ai_model.py train
```
For interferance
```py
python ai_model.py
```
