from torch.nn import Module, CrossEntropyLoss, Linear


class Tagger(Module):
    def __init__(self, n_labels: int, llm_output_size: int):
        super(Tagger, self).__init__()

        # Label POS
        self.ffn = Linear(llm_output_size, n_labels)
        

    def forward(self, x):
        output = self.ffn(x)

        return output
    
