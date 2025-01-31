from torch.utils.data import Dataset

class SDTextDataset(Dataset):
    def __init__(self, anno_path, tokenizer_one, is_sdxl=False, tokenizer_two=None):
        if anno_path.endswith(".txt"):
            self.all_prompts = []
            with open(anno_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue 
                    else:
                        self.all_prompts.append(line)
        else:
            self.all_prompts = pickle.load(open(anno_path, "rb"))
    
        self.all_indices = list(range(len(self.all_prompts)))

        self.is_sdxl = is_sdxl
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

        print(f"Loaded {len(self.all_prompts)} prompts")

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, idx):
        prompt = self.all_prompts[idx]
        if prompt == None:
            prompt = ""


        text_input_ids_one = self.tokenizer_one(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        output_dict = {
            'index': self.all_indices[idx],
            'key': prompt,
            'text_input_ids_one': text_input_ids_one,
        }

        return output_dict 