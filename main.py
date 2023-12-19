import torch
from torch import nn

from transformers import AutoTokenizer, AdamW, HfArgumentParser, get_linear_schedule_with_warmup

# from torch.optim import lr_scheduler

from src.dataloader import get_dataloader
from src.model import ZSSFModel
from config import DataTrainingArguments, ModelArguments, TrainingArguments
from trainer import train, eval
from utils.data_utils import log_params

import sys
import os
import time
import json


def main():
    # parse arguments 
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # For logging
    current_time = time.localtime()
    current_time = f"{current_time.tm_year}_{current_time.tm_mon}_{current_time.tm_mday}_{current_time.tm_hour}_{current_time.tm_min}_{current_time.tm_sec}"

    save_path = f'{training_args.output_dir}/{data_args.target_domain}/Sample{data_args.n_samples}/'
    # log_path = f'{training_args.output_dir}/{data_args.target_domain}/Sample{data_args.n_samples}/'
    # model_path = f'{training_args.output_dir}/model/{data_args.target_domain}/Sample{data_args.n_samples}/'
    log_dict = {}

    log_params(log_dict, [model_args, data_args, training_args])
    # load pretrained roberta-base
    # tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
    # load pretrained BERT and define model 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    model = nn.DataParallel(ZSSFModel(model_args, training_args).cuda()) if torch.cuda.is_available() else ZSSFModel(
        model_args, training_args)

    # # get dataloader
    # dataloader_train, dataloader_val, dataloader_test = get_dataloader(
    #     data_args.target_domain,
    #     training_args.per_device_train_batch_size,
    #     data_args.n_samples,
    #     data_args.dataset_path,
    #     tokenizer,
    #     training_args.use_dep,
    #     training_args.use_pos,
    # )

    # get dataloader
    dataloader_train, dataloader_val, dataloader_test, dataloader_unseen, dataloader_seen = get_dataloader(
        data_args.target_domain,
        training_args.per_device_train_batch_size,
        data_args.n_samples,
        data_args.dataset_path,
        tokenizer,
        training_args.use_dep,
        training_args.use_pos,
    )

    if data_args.run_mode == 'train':
        print("Training mode...")
        # loss function, optimizer, ...
        optim = AdamW(model.parameters(), lr=training_args.learning_rate, correct_bias=True)
        #
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=training_args.warmup_steps,
                                                    num_training_steps=training_args.max_steps)

        # scheduler option => ReduceLROnPlateau
        # scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=10, min_lr=1e-10,
        #                                verbose=True)

        os.makedirs(save_path, exist_ok=True)

        print(f'Target Domain: {data_args.target_domain}\tN Samples: {data_args.n_samples}')

        best_step, best_f1 = train(model=model,
                                   dataloader_train=dataloader_train,
                                   dataloader_val=dataloader_val,
                                   optim=optim,
                                   scheduler=scheduler,
                                   eval_steps=training_args.eval_steps,
                                   total_steps=training_args.max_steps,
                                   early_stopping_patience=data_args.early_stopping_patience,
                                   model_save_path=save_path,
                                   log_dict=log_dict,
                                   tokenizer=tokenizer,
                                   use_dep=training_args.use_dep,
                                   use_pos=training_args.use_pos,
                                   )

        print("Training finished.")
        print(f"Best validation f1 score {best_f1: .2f} at training step {best_step}")

        with open(save_path + 'log.json', 'w') as json_out:
            json.dump(log_dict, json_out, indent=4)


    elif data_args.run_mode == 'test':
        print("Test mode...")
        # Prediction / Test
        model.load_state_dict(torch.load(save_path + f"best-model-parameters.pt"))
        results = eval(model, dataloader_test, data_args.target_domain, tokenizer, save_path + "test_output.json", training_args.use_dep,
        training_args.use_pos)
        print(f"F1 Score at prediction: {results['fb1']}")

        log_dict['test_result'] = results['fb1']


        print("Unseen mode...")
        # Prediction / Unseen test
        # model.load_state_dict(torch.load(save_path + f"best-model-parameters.pt"))
        results = eval(model, dataloader_unseen, data_args.target_domain, tokenizer,
                       save_path + "unseen_output.json", training_args.use_dep,
                       training_args.use_pos)
        print(f"F1 Score at unseen prediction: {results['fb1']}")

        log_dict['unseen_result'] = results['fb1']

        print("Seen mode...")
        # Prediction / Seen test
        # model.load_state_dict(torch.load(save_path + f"best-model-parameters.pt"))
        results = eval(model, dataloader_seen, data_args.target_domain, tokenizer,
                       save_path + "seen_output.json", training_args.use_dep,
                       training_args.use_pos)
        print(f"F1 Score at seen prediction: {results['fb1']}")

        log_dict['seen_result'] = results['fb1']


    else:
        print("Invalid input: option \"run_mode\" got wrong value.")

    return


if __name__ == "__main__":
    main()
