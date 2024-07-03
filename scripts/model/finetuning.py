import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
from .chatbot import MentalHealthChatbot

OUTPUT_DIR = "meta-llama-finetuned"

# Finetune using Qlora
class FineTune:
    def __init__(self, dataset_name="heliosbrahma/mental_health_chatbot_dataset", model_name="meta-llama/Llama-2-70b-chat-hf"):
        self.dataset = load_dataset(dataset_name)

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,            # load model in 4-bit precision
            bnb_4bit_quant_type="nf4",    # pre-trained model should be quantized in 4-bit NF format
            bnb_4bit_use_double_quant=True, # Using double quantization as mentioned in QLoRA paper
            bnb_4bit_compute_dtype=torch.bfloat16, # During computation, pre-trained model should be loaded in BF16 format
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config, # Use bitsandbytes config
            device_map="auto",  # Specifying device_map="auto" so that HF Accelerate will determine which GPU to put each layer of the model on
            trust_remote_code=True, # Set trust_remote_code=True to use meta llama model with custom code
        )
        self.model = prepare_model_for_kbit_training(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token # Setting pad_token same as eos_token

    def finetune(self):
        lora_alpha = 32 # scaling factor for the weight matrices
        lora_dropout = 0.05 # dropout probability of the LoRA layers
        lora_rank = 32 # dimension of the low-rank matrices

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_rank,
            bias="none",  # setting to 'none' for only training weight params instead of biases
            task_type="CAUSAL_LM",
            target_modules=[         # Setting names of modules in falcon-7b model that we want to apply LoRA to
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]
        )

        peft_model = get_peft_model(self.model, peft_config)

        per_device_train_batch_size = 16 # reduce batch size by 2x if out-of-memory error
        gradient_accumulation_steps = 4  # increase gradient accumulation steps by 2x if batch size is reduced
        optim = "paged_adamw_32bit" # activates the paging for better memory management
        save_strategy="steps" # checkpoint save strategy to adopt during training
        save_steps = 10 # number of updates steps before two checkpoint saves
        logging_steps = 10  # number of update steps between two logs if logging_strategy="steps"
        learning_rate = 2e-4  # learning rate for AdamW optimizer
        max_grad_norm = 0.3 # maximum gradient norm (for gradient clipping)
        max_steps = 320        # training will happen for 320 steps
        warmup_ratio = 0.03 # number of steps used for a linear warmup from 0 to learning_rate
        lr_scheduler_type = "cosine"  # learning rate scheduler

        training_arguments = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            bf16=True,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            push_to_hub=True,
        )

        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=self.dataset['train'],
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=1024,
            tokenizer=self.tokenizer,
            args=training_arguments,
        )

        trainer.train()
        trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    bot = MentalHealthChatbot()
