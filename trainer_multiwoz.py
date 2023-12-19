from collections import defaultdict
import os
import torch
from tqdm import tqdm
from itertools import chain, repeat

from utils.data_utils_multiwoz import y1_set
from src.conll2002_metrics import conll2002_measure

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def repeater(dataloader): # for infinite dataloader loop
    for loader in repeat(dataloader):
        for data in loader:
            yield data

def train(model,
            model_save_path,
            dataloader_train, 
            dataloader_val, 
            optim, 
            scheduler, 
            eval_steps, 
            total_steps, 
            early_stopping_patience,
            log_dict,
            tokenizer,
            use_dep,
            use_pos,
            ):

    model.train()
    log_dict['eval_results'] = []

    repeat_dataloader = repeater(dataloader_train) # for infinite loop
    pbar = tqdm(repeat_dataloader, total=total_steps, desc="Start Training")
    
    best_step = 0
    best_f1_score = 0
    patience = 0
    losses = []
    coarse_losses = []

    for i, features in enumerate(pbar):
        if i == total_steps:
            print(f"Training step reached set maximum steps: {total_steps}")
            break

        # optim.zero_grad()
        # features = features.to(device)
        # loss, _ = model(features, optim, scheduler)
        crf_loss, coarse_loss, fine_loss, _ = model(features)
        # loss = loss.mean()
        # crf_loss = crf_loss.mean()
        coarse_loss = coarse_loss.mean()
        coarse_losses.append(coarse_loss.detach().cpu().item())
        # fine_loss = fine_loss.mean()

        loss = crf_loss + 0.05*fine_loss
        loss = loss.mean()
        losses.append(loss.detach().cpu().item())

        # optim.zero_grad()
        # loss.backward(retain_graph=True)
        # # loss.backward()
        # losses.append(loss.detach().cpu().item())
        #
        # optim.step()
        # scheduler.step()
        # # scheduler.step(loss)


        # first step
        optim.zero_grad()
        coarse_loss.backward(retain_graph=True)
        optim.step()
        scheduler.step()
        # scheduler.step(coarse_loss)

        # second step
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        scheduler.step()
        # scheduler.step(loss)

        # pbar.set_description(f"LOSS: {losses[-1]:.4f}")
        pbar.set_description(f"COARSE_LOSS: {coarse_losses[-1]:.4f}  LOSS: {losses[-1]:.4f}")

        # evaluation
        if (i + 1) % eval_steps == 0:
            result = {}
            eval_f1 = eval(model, dataloader_val, tokenizer=tokenizer, use_dep=use_dep, use_pos=use_pos)['fb1']
            print(f"Result(F1-Score) at step {i+1}: {eval_f1}")
            result['step'] = i + 1
            result['f1-score'] = eval_f1
            log_dict['eval_results'].append(result)
            # scheduler => f1-score
            # scheduler.step(eval_f1)

            if eval_f1 > best_f1_score:
                """
                when better evaluation f1 score is found:
                update best_f1_score and best_step
                & save model's parameter
                """
                print("Found better model!")

                os.makedirs(model_save_path, exist_ok=True)
                if os.path.isfile(model_save_path+f'best-model-parameters.pt'):
                    os.remove(model_save_path+f'best-model-parameters.pt')
                torch.save(model.state_dict(), model_save_path+f'best-model-parameters.pt')
                best_f1_score = eval_f1
                best_step = i
                patience = 0

            else:
                patience += 1
                if patience == early_stopping_patience:
                    print(f"Early stop at step {i+1}")
                    i += 1
                    break

            model.train()

    log_dict['stopped_step'] = i
    log_dict['eval_best_step'] = best_step
    log_dict['eval_best_f1_score'] = best_f1_score

    return best_step, best_f1_score

def eval(model, dataloader, tgt_domain=None, tokenizer=None, out_file=None,  use_dep=False, use_pos=False):
    """
    evalutation function for validation dataset and test dataset
    
    returns
    ----------
    List of losses, F1 Score
    """
    model.eval()
    losses = []
    total_preds = []
    total_targets = []
    out_dict = defaultdict(dict)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for features in pbar:
            # _loss, logits = model(features, optim, scheduler)
            _loss, coarse_loss, fine_loss, logits = model(features)
            
            loss = _loss.mean()
            losses.append(loss.detach().cpu().item())

            # pred = torch.argmax(logits, dim=2)
            # with gcl logits
            pred = torch.argmax(logits[0], dim=2)
            true_labels = features['labels']
            
            total_preds.extend(pred.tolist())
            total_targets.extend(true_labels.tolist())
            
            remove_char = ['[CLS]', ' ']
            for input_id, p in zip(features['input_ids'], pred):
                if use_dep:
                    slot_dep_utter = tokenizer.decode(input_id).split(' [SEP] ')
                    for rc in remove_char:
                        slot_dep_utter[0] = slot_dep_utter[0].replace(rc, '')

                    slot = tokenizer.decode(input_id[(p != 0)])
                    try:
                        slot_type, utterance_dep, utterance, _ = slot_dep_utter
                    except:
                        slot_type, utterance_dep, utterance = slot_dep_utter
                        utterance = utterance.split(' [SEP] ')[0]

                    out_dict[utterance_dep+' -> '+utterance][slot_type] = slot if slot != '' else 'NONE'
                else:
                    slot_utter = tokenizer.decode(input_id).split(' [SEP] ')
                    for rc in remove_char:
                        slot_utter[0] = slot_utter[0].replace(rc, '')
                
                    slot = tokenizer.decode(input_id[(p != 0)])
                    try:
                        slot_type, utterance, _ = slot_utter
                    except:
                        slot_type, utterance = slot_utter
                        utterance = utterance.split(' [SEP] ')[0]
                    out_dict[utterance][slot_type] = slot if slot != '' else 'NONE'

        # below is for check
        # rand = torch.randint(query.size()[0], (1,)).item()
        # decoded = tokenizer.decode(query[rand])

        # print("Query     : ", decoded)
        # print("Answer    : ", targets[rand])
        # print("Prediction: ", pred[rand])
        total_targets = list(chain.from_iterable(total_targets))
        total_preds = list(chain.from_iterable(total_preds))
        total_lines = []
        for target, pred in zip(total_targets, total_preds):
            bin_target = y1_set[target]
            bin_pred = y1_set[pred]

            total_lines.append("w" + " " + bin_pred + " " + bin_target)

        result = conll2002_measure(total_lines)
        if out_file is not None:
            import json
            with open(out_file, "w") as json_file:
                json.dump(out_dict, json_file, indent=4)

    return result