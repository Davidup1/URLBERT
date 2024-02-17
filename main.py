import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import options
from dataloader import generate_dataloader
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
)
import random
import numpy as np
from timerecord import format_time
from tqdm.notebook import tqdm
from torch.optim import AdamW
import time
from AL import virtual_adversarial_training
from DropAL import dropAlloss
from torch.cuda.amp import autocast as autocast, GradScaler


class Config:
    def __init__(self):
        pass

    def training_config(
            self,
            batch_size,
            epochs,
            learning_rate,
            weight_decay,
            device,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device


def train(model, train_dataloader, val_dataloader, train_sampler, val_sampler, config):
    """
    :param model: nn.Module
    :param train_dataloader: DataLoader
    :param config: Config
    """

    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_con = AdamW(model.parameters(), lr=config.learning_rate * 10, weight_decay=config.weight_decay)

    time_t0 = time.time()
    epoch = 0
    scaler = GradScaler()
    for cur_epc in tqdm(range(int(config.epochs)), desc="Epoch"):
        epoch += 1
        training_loss = []
        evaluation_loss = []
        print("Epoch: {}".format(cur_epc+1))
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc='Step')):
            train_sampler.set_epoch(step)
            val_sampler.set_epoch(step)
            input_ids, attention_mask, token_type_ids, labels = batch[0].long().to(config.device), \
                batch[1].long().to(config.device), batch[2].long().to(config.device), batch[3].long().to(config.device)
            with autocast():
                output = model(input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               labels=labels,
                               output_hidden_states=True,
                               )
                loss = output[0]
                logits = output[1]
                hidden_states = output[2][0]
                adv_loss = virtual_adversarial_training(model, hidden_states, token_type_ids, attention_mask, logits)
                if adv_loss:
                    loss = adv_loss * 10 + loss
                else:
                    pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with autocast():
                contrastive_loss = dropAlloss(model, input_ids, token_type_ids, attention_mask, config.batch_size, config.device)
                # loss = loss + contrastive_loss
            optimizer_con.zero_grad()
            scaler.scale(contrastive_loss).backward()
            scaler.step(optimizer_con)
            scaler.update()
            model.zero_grad()
            loss = loss + contrastive_loss
            training_loss.append(loss.item())

            if (step+1) % 500 == 0:
                print("Step {} training loss: {}".format(step+1, sum(training_loss)/len(training_loss)))
        train_loss = sum(training_loss)/len(training_loss)


        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_dataloader, desc='Step')):
                input_ids, attention_mask, token_type_ids, labels = batch[0].long().to(config.device), \
                    batch[1].long().to(config.device), batch[2].long().to(config.device), batch[3].long().to(config.device)
                with autocast():
                    output = model(input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   labels=labels,
                                   output_hidden_states=True,
                                   )
                    loss = output[0]
                evaluation_loss.append(loss.item())

                if (step+1) % 500 == 0:
                    print("Step {} evaluation loss: {}".format(step+1, sum(evaluation_loss)/len(evaluation_loss)))

        eval_loss = sum(evaluation_loss)/len(evaluation_loss)
        time_t1 = time.time()
        cost_time = format_time(time_t1 - time_t0)

        print("Training loss: {}; Evaluation loss: {} ; Epoch{} cost time: {}".format(train_loss, eval_loss, epoch, cost_time))


def main(args):
    batch_size = args.batch_size
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        dist.barrier()
        ddp = True
    else:
        ddp = False

    input_ids_train = torch.load("./tokenized_data/train_input_ids.pt")
    attention_masks_train = torch.load("./tokenized_data/train_attention_mask.pt")
    token_type_ids_train = torch.load("./tokenized_data/train_token_type_ids.pt")
    labels_train = torch.load("./tokenized_data/train_labels.pt")

    input_ids_val = torch.load("./tokenized_data/test_input_ids.pt")
    attention_masks_val = torch.load("./tokenized_data/test_attention_mask.pt")
    token_type_ids_val = torch.load("./tokenized_data/test_token_type_ids.pt")
    labels_val = torch.load("./tokenized_data/test_labels.pt")

    vocab_size = 5000

    config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "hidden_dropout_prob": 0.2,
        "vocab_size": vocab_size
    }

    config = AutoConfig.from_pretrained('./bert_config/', **config_kwargs)

    model = AutoModelForMaskedLM.from_config(
        config=config,
    )
    model.resize_token_embeddings(vocab_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    input_ids_train = torch.cat(input_ids_train, dim=0)
    attention_masks_train = torch.cat(attention_masks_train, dim=0)
    token_type_ids_train = torch.cat(token_type_ids_train, dim=0)
    labels_train = torch.cat(labels_train, dim=0)

    input_ids_val = torch.cat(input_ids_val, dim=0)
    attention_masks_val = torch.cat(attention_masks_val, dim=0)
    token_type_ids_val = torch.cat(token_type_ids_val, dim=0)
    labels_val = torch.cat(labels_val, dim=0)

    model.cuda(args.local_rank)
    if ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        gpu_num = torch.distributed.get_world_size()
    else:
        gpu_num = 1

    train_list = [input_ids_train, attention_masks_train, token_type_ids_train, labels_train]
    val_list = [input_ids_val, attention_masks_val, token_type_ids_val, labels_val]
    train_dataloader, val_dataloader, train_sampler, val_sampler = generate_dataloader(train_list, val_list, ddp, batch_size*gpu_num)

    config = Config()
    config.training_config(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr * gpu_num, weight_decay=args.weight_decay, device=device)

    train(model, train_dataloader, val_dataloader, train_sampler, val_sampler, config)

    torch.save(model, 'bert_model/small/urlBERT.pt')

if __name__ == "__main__":

    seed_val = 2024
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    args = options.args_parser()
    main(args)
