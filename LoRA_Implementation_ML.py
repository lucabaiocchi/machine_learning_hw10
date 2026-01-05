# -*- coding: utf-8 -*-
"""
# Homework 10

NOTE: All code inside the "MY CODE - END OF MY CODE" section has been written by me, Luca Baiocchi. Instructions of the assignment and provided helper code has been
added as required to ensure code is reproducible. 

Pre-trained large language models (LLMs) compress knowledge from vast amounts of text, but in practice they often need additional fine-tuning to perform well 
on specific downstream tasks. Fine-tuning can be done by updating all of a model’s parameters (full-model fine-tuning) or by modifying only a small subset, 
a strategy known as parameter-efficient fine-tuning (PEFT), which is typically preferred for its computational and memory savings.

Low-Rank Adaptation (LoRA) is a popular option for PEFT, because it introduces small trainable matrices (A and B) that adjust a model’s 
generation without altering most existing parameters. In this homework, we will fine-tune a pretrained LLM, HuggingFace’s SmolLM2,
 on a downstream dataset using both full-model training and LoRA. After evaluating the full-model version in Part 1, we will implement 
 our own LoRA-based fine-tuning in Part 2. For more background on LoRA, you may find it helpful to review the original paper: https://arxiv.org/pdf/2106.09685


"""

import torch
import numpy as np
import torch.nn as nn
from typing import List

if __name__ == '__main__':

    #!pip install datasets # comment out when submitting
    #!pip install evaluate # comment out when submitting

    import os
    import torch
    import random
    import evaluate
    import requests
    import numpy as np
    import pandas as pd
    import torch.nn as nn
    from io import BytesIO
    from typing import List

    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import get_scheduler
    from transformers import DataCollatorWithPadding
    from datasets import load_dataset
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, login



if __name__ == '__main__':
    checkpoint = "HuggingFaceTB/SmolLM2-135M"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = '<|pad_token|>'
    print(tokenizer.eos_token)
    print(tokenizer.bos_token)
    print(tokenizer.pad_token)

    model_full = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    print(model_full)

    inputs = tokenizer.encode("Gravity is", return_tensors="pt").to(device)
    outputs = model_full.generate(inputs)
    print(tokenizer.decode(outputs[0]))

######################## DO NOT MODIFY ########################
# Download the dataset
if __name__ == '__main__':
    ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")
    print(ds['train'][0])

############ paste your HuggingFace access token here! ############
if __name__ == '__main__':
    login()


"""# Part 1: Full-model fine-tuning on instruction dataset (2 point)

Before we implement the LoRA fine-tuning workflow, we will first establish a baseline using **full-model fine-tuning**. During instruction fine-tuning, it is common to apply a template to the dataset to format inputs and outputs into a consistent conversational structure that the model can learn from. This templated dataset is then used to fine-tune the LLM.
The base-model is trained to predict the next token and may not inherently understand what a “user message” or an “assistant response” is. Templates impose a conversation structure to the text, from whcih the model is trained to predict when the assistant's turn is and what style of output to produce. For instance:

```
<|user|>: What is 1+1?
<|assistant|>: It is 2.
```

In this section, you will write your own functions required to fine-tune the model (e.g. processing the original dataset, creating the training loop or using a Trainer), similar to what you have implemented in previous assignments.
We will fine-tune the model on the *everydata-conversation* split of the SmolTalk dataset, which contains multi-turn conversations between a user and an assistant across a wide range of topics (sports, fashion, health, entertainment, and more). After fine-tuning, you will see how even a simple instruction-tuning approach can significantly improve generation quality.

## 1.a Convert the dataset to chat template (0.5 point)

**TODO**
* Complete the apply_my_chat_template() function to convert a raw text data into a chat-templated format. You can define your own chat template, or look up any common chat templates used in practice.

Given a list of user-assistant messages, your function should convert it into a single formatted string. For instance:
```
user_input = [{'content': 'how are you?', 'role': 'user'},
              {'content': 'I'm good, thanks! And you?', 'role': 'assistant'}
]
templated_output = apply_my_chat_template(user_input)
print(templated_output)
-----------------------------------------
<|user|> How are you?
<|assistant|> I'm good, thanks! And you?
-----------------------------------------
```
"""
##### MY CODE #######
def apply_my_chat_template(messages: List) -> str:

    formatted = ""

    # given a list user-assistant interactions, create a formatted string
    #       with "user" and "assistant" headers
    for message in messages:
      # User
      if message['role'] == "user":
        formatted += '<|user|> ' + message['content']
      # Assistant
      else:
        formatted += '<|assistant|> ' + message['content']
    # We want to add a \n every time we switch
      formatted += '\n'


    return formatted


if __name__ == '__main__':
  
    formatted_input = apply_my_chat_template([{'content': 'hi there!', 'role': 'user'},
                                              {'content': 'hi, how is it going?', 'role':'assistant'}])
    print(formatted_input)

##### END OF MY CODE #######

"""## 1.b Fine-tuning the model (1.5 point)

Unlike LLMs with a classification head, which output one of a fixed set of labels (such as the model used in Homework 8), SmolLM2 generates open-ended text responses. There are multiple ways to evaluate the quality of such generative models. In this assignment we will use the validation loss after fine-tuning as the quantitative measure of model quality. In addition, you will design a set of custom prompts to qualitatively compare the model’s responses before and after fine-tuning.

Your fine-tuned model will be evaluated using the check_validation_loss() function. The function takes your fine-tuned model, apply_my_template() function, and the original dataset.

**TODO**
* Define your **custom prompts (10 total)** used evaluate the model generation quality.
* Complete the **count_trainable_parameter()** function to count the number of tunable parameters.
* Set up your own **training function** to run training. Achieve validation loss < 1.
"""
##### MY CODE #######
def check_validation_loss(model, tokenizer, my_template_function, dataset):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenized_dataset = dataset.map(lambda x: tokenizer(my_template_function(x['messages'])))
    tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x['input_ids']})

    tokenized_dataset = tokenized_dataset.select_columns(['input_ids','labels','attention_mask'])
    tokenized_dataset.set_format("torch")

    valid_dataloader = DataLoader(tokenized_dataset['test'],
                                  batch_size=1)

    model.to(device)
    batch_loss = []
    with torch.no_grad():
        for batch in valid_dataloader:
            out = model(**{k:v.to(device) for k, v in batch.items()})
            batch_loss.append(out.loss)

    avg_loss = sum(batch_loss) / len(batch_loss)

    return avg_loss

def count_trainable_parameters(model):
    
    # Counting the total number of tunable parameters

    parameter_count = 0
    for par in model.parameters():
      if par.requires_grad: # We only want tunable parameters
        parameter_count += par.numel()

    return parameter_count

##### END OF MY CODE #######

########################### MY PROMPTS ###########################
# TODO: define 10 user prompts to assess the model quality
YOUR_PROMPTS = [
                [{'role':'user', 'content': 'Why is the sky blue?'}], 
                [{'role':'user', 'content': 'What is 5+5?'}],
                [{'role':'user', 'content': 'Would you rather be a city mouse or a country mouse?'}],
                [{'role':'user', 'content': 'In a 4 stop intersection, who has the right of way?'}],
                [{'role':'user', 'content': 'Who was the first president of the United States?'}],
                [{'role':'user', 'content': 'How many "r" are there in the word "strawberry"?'}],
                [{'role':'user', 'content': 'Tell me any story you would like.'}],
                [{'role':'user', 'content': 'Why are racoons common in cities?'}],
                [{'role':'user', 'content': 'What are the main ingredients of pizza?'}],
                [{'role':'user', 'content': 'What is the most important rule in soccer?'}],
               ]

"""Here, write any training code necessary to run full model fine-tuning of SmolLM2 on the instruction dataset. Make sure to put your code block under `if __name__ == '__main__': ` to avoid having the code running during the autograding."""


##### MY CODE #######

if __name__ == '__main__':
    #Do Training Here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenized_dataset = ds.map(lambda x: tokenizer(apply_my_chat_template(x['messages'])))
    tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x['input_ids']})

    tokenized_dataset = tokenized_dataset.select_columns(['input_ids','labels','attention_mask'])
    tokenized_dataset.set_format("torch")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,return_tensors="pt"
    )

    metric = evaluate.load("accuracy")
    train_dataloader = DataLoader(tokenized_dataset['train'],
                                  batch_size=1)
    model_full.to(device)
    # Set up optimizer, here we do it on the full list of parameters
    optimizer = AdamW(model_full.parameters(), lr=5e-5)

    model_full.train()
    for batch in train_dataloader:
      model_full.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)

      # Forward pass
      out = model_full(**{k:v.to(device) for k, v in batch.items()})
      loss = out.loss

      # Backward pass
      loss.backward()
      optimizer.step()


    #replace_target_with_lora_layer(model_full, r=8, alpha=16)
    model_full.eval()
    #test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=5)


    full_valid_loss = check_validation_loss(model_full, tokenizer, apply_my_chat_template, ds)
    print(full_valid_loss)


if __name__ == '__main__':
    for idx, p in enumerate(YOUR_PROMPTS):
        templated_ipnut = apply_my_chat_template(p) + '\n<|im_start|><|assistant|>: '
        tokenized_input = tokenizer(templated_ipnut, return_tensors='pt')
        out = model_full.generate(**{k:v.to(device) for k, v in tokenized_input.items()}, max_new_tokens=50)

        print(f"{idx}")
        print(">>> Raw input text: ", p)
        print(">>> Templated output: ", templated_ipnut)
        print(">>> Model full output: \n" , tokenizer.decode(out[0]))
        print('\n')


##### END OF MY CODE #######

############### TO SUBMIT ###############
YOUR_HF_USER_NAME = "" # Removed for privacy
YOUR_FULL_MODEL_REPO_NAME = "" # Removed for privacy
#########################################

######### SAVE THE MODEL WEIGHTS #########
if __name__ == '__main__':

    # first, save locally
    model_full.save_pretrained(f"{MODEL_SAVE_PATH}/{YOUR_FULL_MODEL_REPO_NAME}")
    tokenizer.save_pretrained(f"{MODEL_SAVE_PATH}/{YOUR_FULL_MODEL_REPO_NAME}")

    # then upload to huggingface
    api = HfApi()
    api.create_repo(YOUR_FULL_MODEL_REPO_NAME, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=f"{MODEL_SAVE_PATH}/{YOUR_FULL_MODEL_REPO_NAME}",
        repo_id=f"{YOUR_HF_USER_NAME}/{YOUR_FULL_MODEL_REPO_NAME}"
    )

"""

# Part 2: Motivating low-rank adaptation (LoRA) training (0.5 point)

When we fine-tune a pretrained model, we are effectively learning an update to the model’s weight matrices:
$$W' = W + \Delta W$$
where $W$ is the pretrained weights (frozen) and $\Delta W $ is the learned task-specific update.

Instead of learning a full matrix $\Delta W \in \mathbb{R}^{d \times k}$,
LoRA assumes that this update has low intrinsic rank.
That is, we can approximate:

$$\Delta W = B A$$

where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, and $r \ll \min(d, k)$ (the *rank* of the update).
The model’s forward pass becomes:
   $
   y = (W + s \cdot B A) x
   $,
where $s$ is a scaling factor ($\alpha$ / r) that balances the LoRA update magnitude.

In this section, you will check if this assumption indeed holds by analyzing the $\Delta W$ between the base model and its fine-tuned version.

**TODO**
* Implement the reconstruction_err() function, which computes the reconstruction error of a matrix when it is approximated using its top-rank components. This helps us analyze whether the LoRA's low-rank assumption always holds. If it does not, what does this implies for the performance of LoRA fine-tuning?
"""

################## DO NOT CHANGE ##################
def plot_reconstruction_error(df):
    ncols=3
    nrows=2
    f, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,6))

    num_lines = 30
    cmap = cm.get_cmap("viridis")

    for idx, target_module in enumerate(['k_proj','q_proj','o_proj','v_proj','up_proj','gate_proj']):
        for i, col in enumerate([c for c in df.columns if target_module in c]):
            color = cmap(i / (num_lines - 1))
            axes[idx//3][idx%3].plot(df['rank'], df[col], color=color, label=col, marker='.')
        axes[idx//3][idx%3].set_title(target_module)

    plt.show()


##### MY CODE #######
def reconstruction_err(W, r):

    reconstruction_error = None
 
    # TODO:
    # 1. Perform SVD on the matrix W and compute its best rank-r approximation
    #    by reconstructing W' using the top-r singular values and corresponding vectors.
    U, S, V = torch.linalg.svd(W, full_matrices=False)
    actual_rank = torch.sum(S>0).item()

    r = min(r, actual_rank) # We use the actual_rank if r is bigger


    # 2. Return the reconstruction error as the Frobenius norm between W and W', ||W - W'||_F
    #Components that are top-rank
    top_U = U[:, :r]
    top_S = S[:r]
    top_V = V[:r, :]
    reconstructed_matrix = (top_U * top_S) @ top_V
    reconstruction_error = torch.norm(W - reconstructed_matrix, p='fro')

    return reconstruction_error

##### END OF MY CODE #######

"""We are now going to load the base model and the model you have just fine-tuned to compute the $\Delta W$ between the two models. Then, you will analyze whether $\Delta W$ indeed has low intrinsic dimensionality by

1. computing the reconstruction error of $\Delta W$ when it is approximated using its top-r components, and
2. plotting how the error converges towards 0 as r increases.

"""

if __name__ == '__main__':

    # load the base model and the model we trained!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model1 = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model2 = AutoModelForCausalLM.from_pretrained(f"{YOUR_HF_USER_NAME}/{YOUR_FULL_MODEL_REPO_NAME}")

################ VISUALIZATION CODE (could take a few minutes to run) ################
if __name__ == '__main__':
    # To visualize the reconstruction error
    d = {}
    for name, p in model1.named_parameters():
        if 'self_attn' in name or 'mlp' in name:
            w1 = model1.state_dict()[name]
            w2 = model2.state_dict()[name]
            d[name] = {}
            for r in [1, 8, 32, 64, 128, 256, 576]:
                w_delta = w1 - w2
                d[name][r] = reconstruction_err(w_delta, r)

    # create a dataframe
    df = pd.DataFrame(d).reset_index().rename(columns={'index':'rank'})

    # plot; earlier layers (close to 0) have darker colors
    plot_reconstruction_error(df)

"""# Part 3: LoRA fine-tuning (4 points)


In this section, you will implement parameter-efficient training using LoRA. You will define a custom LoRA layer and attach it to each target module in the model. During training, only the LoRA parameters will be updated while the original model weights remain frozen. The choice of rank (*r*) plays an important role in both performance and efficiency.

How does the number of trainable parameters change as we vary the rank? Is the model performance affected? To explore this trade-off, you can perform a hyperparameter sweep over different values of *r* to determine which setting offers the best balance between performance and efficiency.

## 3.a Create a LoRA model by injecting a custom LoRA layer (1 point)


When we fine-tune a pretrained model, we are effectively learning an update to the model’s weight matrices:
$$W' = W + \Delta W$$
where $W$ is the pretrained weights (frozen) and $\Delta W $ is the learned task-specific update.

Instead of learning a full matrix $\Delta W \in \mathbb{R}^{d \times k}$,
LoRA assumes that this update has low intrinsic rank.
That is, we can approximate:

$$\Delta W = B A$$

where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, and $r \ll \min(d, k)$ (the *rank* of the update).
The model’s forward pass becomes:
   $
   y = (W + s \cdot B A) x
   $,
where $s$ is a scaling factor ($\alpha$ / r) that balances the LoRA update magnitude.


In this section, define a custom linear layer for LoRA. It should 1. initialize the A and B matrices, and 2. implement a forward pass that applies the LoRA update to the original layer output.


**TODO**
* Complete the LoRALinear class, which creates low-rank A and B matrices and  added the LoRA weights applied output to the original linear layer’s output.
"""
##### MY CODE #######

if __name__ == '__main__':
    checkpoint = "HuggingFaceTB/SmolLM2-135M"
    model_lora  = AutoModelForCausalLM.from_pretrained(checkpoint)

class LoRALinear(nn.Module):
    def __init__(self,
                 base_layer: nn.Linear,
                 in_features: int,
                 out_features: int,
                 r: int, alpha: int):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        self.A = nn.Parameter(torch.rand(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
     

    def forward(self, x):
        # NOTE: LoRALinear(x) shape should be the same as
        #       base_layer(x) shape
       
        base_layer_result = self.base_layer(x)
        new_matrix = self.scaling * (self.B @ self.A)
        lora_out = x @ new_matrix.t()

        return lora_out + base_layer_result

##### END OF MY CODE #######

"""## 3.b Inject the custom layer into the base model (1 point)

Inject the custom LoRA layer into the base model by iterating over its modules and replacing each target linear layer with a LoRALinear module.

**TODO**
* Complete **replace_target_with_lora_layer()** to  locate each target module and replace it with a LoRALinear instance containing the LoRA A and B matrices.
"""
##### MY CODE #######
def replace_target_with_lora_layer(model,
                                   r: int,
                                   alpha: int,
                                   target_modules=['q_proj','k_proj','v_proj','o_proj', ]) -> None:

    modules_to_replace = []
    for name, m in model.named_modules():
        layer_name = name.split('.')[-1]
        if layer_name in target_modules:
          
            modules_to_replace.append((name, m))
            # Find the parent modules
    #print(modules_to_replace)
    for name, m in modules_to_replace:
        parent_module = model

        layer_path = name.split('.')
        layer_name = layer_path[-1]

        for attr in layer_path[:-1]:
          #print(attr)
          if attr.isdigit():
            parent_module = parent_module[int(attr)]
          else:
            parent_module = getattr(parent_module, attr)
      
        # Establish layer
        your_LoRALinear = LoRALinear(
            base_layer=m,
            in_features=m.in_features,
            out_features=m.out_features,
            r=r,
            alpha=alpha
        )
        
        setattr(parent_module, layer_name, your_LoRALinear)

##### END OF MY CODE #######

"""

## 3.c Freeze the base model weights for parameter-efficient fine-tuning (0.5 point)

Finally, ensure that only the LoRA A and B matrices remain trainable by freezing all other model parameters.

**TODO**
* Complete **set_requires_grad_for_lora()** so that only the LoRA parameters require gradients, while all base model weights are frozen.
"""
##### MY CODE #######
def set_requires_grad_for_lora(model):

    for name, p in model.named_parameters():
       
        if "A" in name:
          p.requires_grad = True
        elif "B" in name:
          p.requires_grad = True
        else:
          p.requires_grad = False

##### END OF MY CODE #######

"""## 3.d Train LoRA model (1.5 point)

Fine-tune the LoRA model on the dataset. Do the same learning rate and number of epochs used for full-model fine-tuning still work here? If not, consider varying the learning rates and LoRA ranks to find the best configuration.

**TODO**
* **Fine-tune the LoRA model** on the everyday-chat dataset. Achieve validation loss < 1.
* Compare the LoRA-fine-tuned model with the full fine-tuned model. Which one achieves lower validation loss? Which produces better generations?
* After fine-tuning, upload the adapters to HuggingFace.
"""
##### MY CODE #######
LORA_HYPERPARM = {
    'rank': 256,
    'alpha': 128,
    'target_modules':['q_proj','k_proj','v_proj','o_proj', 'gate_proj', 'down_proj', 'up_proj']
}


if __name__ == '__main__':
    print("Before setting the LoRA: ", count_trainable_parameters(model_lora))
    replace_target_with_lora_layer(model_lora,
                                   r=LORA_HYPERPARM['rank'],
                                   alpha=LORA_HYPERPARM['alpha'],
                                   target_modules=LORA_HYPERPARM['target_modules'],

                                   )
    set_requires_grad_for_lora(model_lora)
    print("After setting the LoRA: ", count_trainable_parameters(model_lora))
    model_lora.to(device)


if __name__ == '__main__':

    #Do Training Here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenized_dataset = ds.map(lambda x: tokenizer(apply_my_chat_template(x['messages'])))
    tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x['input_ids']})

    tokenized_dataset = tokenized_dataset.select_columns(['input_ids','labels','attention_mask'])
    tokenized_dataset.set_format("torch")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,return_tensors="pt"
    )

    metric = evaluate.load("accuracy")
    train_dataloader = DataLoader(tokenized_dataset['train'],
                                  batch_size=1)
    model_lora.to(device)
    # Set up optimizer, here we do it on the full list of parameters
    optimizer = AdamW(model_lora.parameters(),lr=0.0002)
    # Maybe get some epochs going here?
    model_lora.train()
    max_epochs = 2
    for epoch in range(max_epochs):
      for batch in train_dataloader:
        model_lora.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        out = model_lora(**{k:v.to(device) for k, v in batch.items()})
        loss = out.loss

        # Backward pass
        loss.backward()
        optimizer.step()


    #replace_target_with_lora_layer(model_full, r=8, alpha=16)
    model_lora.eval()
    # Recheck Loss
    #final_val_loss = check_validation_loss(model_lora, tokenizer, apply_my_chat_template, ds)
    #print("Validation loss: ", final_val_loss)

    for idx, p in enumerate(YOUR_PROMPTS):
        templated_ipnut = apply_my_chat_template(p) + '\n<|im_start|><|assistant|>: '
        tokenized_input = tokenizer(templated_ipnut, return_tensors='pt')
        out = model_lora.generate(**{k:v.to(device) for k, v in tokenized_input.items()},
                                  max_new_tokens=50,
                                  eos_token_id=tokenizer.eos_token_id)
        print(f"{idx}")
        print(">>> Raw input text: ", p)
        print(">>> Templated output: ", templated_ipnut)
        print(">>> Model full output: \n" , tokenizer.decode(out[0]))
        print('\n')

##### END OF MY CODE #######


############### TO SUBMIT ###############
YOUR_LORA_ADAPTER_REPO_NAME = "" # Redacted for privacy
#########################################

"""Again, write any training code necessary to run the model fine-tuning. Make sure to put your code block under `if __name__ == '__main__': ` to avoid having the code running during the autograding."""

######### SAVE THE LORA ADAPTER WEIGHTS ONLY AND UPLOAD TO HUB #########
if __name__ == '__main__':

    # first, save the adapter (trained A,B matrices) and the tokenizer locally
    adapters = {k: v.cpu() for k, v in model_lora.state_dict().items() if ".A" in k or ".B" in k}
    os.makedirs(f"{MODEL_SAVE_PATH}/{YOUR_LORA_ADAPTER_REPO_NAME}", exist_ok=True)
    torch.save(adapters, f"{MODEL_SAVE_PATH}/{YOUR_LORA_ADAPTER_REPO_NAME}/lora_adapters.pt")
    tokenizer.save_pretrained(f"{MODEL_SAVE_PATH}/{YOUR_LORA_ADAPTER_REPO_NAME}")

    # then push the folder to hub
    api = HfApi()
    api.create_repo(YOUR_LORA_ADAPTER_REPO_NAME, exist_ok=True, repo_type='model')
    api.upload_folder(
        folder_path = f'{MODEL_SAVE_PATH}/{YOUR_LORA_ADAPTER_REPO_NAME}',
        repo_id=f'{YOUR_HF_USER_NAME}/{YOUR_LORA_ADAPTER_REPO_NAME}'
    )

"""To load your trained adapter to the base model, we can reconstruct the model by:"""

if __name__ == '__main__':
    checkpoint = "HuggingFaceTB/SmolLM2-135M"
    model_loaded  = AutoModelForCausalLM.from_pretrained(checkpoint)

    # attach default lora adapters
    replace_target_with_lora_layer(model_loaded,
                                   r=LORA_HYPERPARM['rank'],
                                   alpha=LORA_HYPERPARM['alpha'],
                                   target_modules=LORA_HYPERPARM['target_modules'],
                                   )

    # load the trained adapter from Hub
    url = hf_hub_url(repo_id=f"{YOUR_HF_USER_NAME}/{YOUR_LORA_ADAPTER_REPO_NAME}",
                    filename="lora_adapters.pt")
    tokenizer_loaded = AutoTokenizer.from_pretrained(f"{YOUR_HF_USER_NAME}/{YOUR_LORA_ADAPTER_REPO_NAME}")
    response = requests.get(url)
    response.raise_for_status()
    adapter_bytes = BytesIO(response.content)
    adapter_state_dict = torch.load(adapter_bytes)

    # load the trained adapter to the model, replacing the default weights
    model_loaded.load_state_dict(adapter_state_dict, strict=False)

    # check the validation loss
    valid_loss_loaded = check_validation_loss(model_loaded, tokenizer_loaded, apply_my_chat_template, ds)
    print(valid_loss_loaded)