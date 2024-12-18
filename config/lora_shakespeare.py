import time

dataset = 'shakespeare'
init_from = 'gpt2-medium'

# LoRA parameters
lora_rank = 1024 
lora_alpha = 1024 * 2 
lora_dropout = 0
use_vera = True
use_mlp = True

out_dir = f'out-lora-shakespeare-{init_from}-{lora_rank}-{lora_alpha}'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 2e-4
decay_lr = False
# weight decay
# weight_decay = 1e-1

device = 'cuda'
compile = False
compute_grad_memory = True

