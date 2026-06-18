import transformers, torch

class EN2JAModel:
    def __init__(self):
        self.checkpoint_path = './qwen_model/checkpoint-8800'
        self.pipe = transformers.pipeline(
            "text-generation",
            model=self.checkpoint_path,
            device=0,
            torch_dtype=torch.bfloat16,
        )
    
    def translate(self, txt):
        outputs = self.pipe(
            [{"role": "user", "content": f"EN2JA: {txt}"}],
            tokenizer_encode_kwargs={"enable_thinking": False},
            clean_up_tokenization_spaces=False,
        )

        output = outputs[0]["generated_text"][-1]["content"]
        print(output)

if __name__ == "__main__":
    model = EN2JAModel()
    while True:
        model.translate(input('> '))
