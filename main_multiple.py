import torch
import options
from dataloader import generate_dataloader
from buildmodel import buildBERT
import random
import numpy as np
from timerecord import format_time
from tqdm import tqdm
from torch.optim import AdamW
import time
from AL import virtual_adversarial_training
from DropAL import dropAlloss
from torch.cuda.amp import autocast as autocast, GradScaler
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def search_path(file, directory):
    count = 0
    for filename in os.listdir(directory):
        if file in filename:
            count += 1
    return count


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


def train(model, train_dataloader, train_sampler, config, optimizer, optimizer_con, scaler, data_step):
    """
    :param model: nn.Module
    :param train_dataloader: DataLoader
    :param config: Config
    """

    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")

    time_t0 = time.time()
    training_loss = []

    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc='Step')):
        train_sampler.set_epoch(step)
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
                loss = adv_loss + 10 * loss
            else:
                pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2)
        scaler.step(optimizer)
        scaler.update()

        with autocast():
            contrastive_loss = dropAlloss(model, input_ids, token_type_ids, attention_mask, config.batch_size, config.device)
        optimizer_con.zero_grad()
        scaler.scale(contrastive_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2)
        scaler.step(optimizer_con)
        scaler.update()

        model.zero_grad()
        loss = loss + contrastive_loss
        training_loss.append(loss.item())

        if (step+1) % 500 == 0:
            print("Step {} training loss: {} adv loss:{} contrastive loss :{}".format(step+1, training_loss[-1],adv_loss.item(), contrastive_loss.item()))
    train_loss = sum(training_loss)/len(training_loss)

    time_t1 = time.time()
    cost_time = format_time(time_t1 - time_t0)

    print("Training loss: {}; Data iteration{} cost time: {}".format(train_loss, data_step, cost_time))


def evaluate(model, val_dataloader, val_sampler, config, data_step):
    time_t0 = time.time()
    evaluation_loss = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader, desc='Step')):
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
            evaluation_loss.append(loss.item())

            if (step+1) % 500 == 0:
                print("Step {} evaluation loss: {}".format(step+1, sum(evaluation_loss)/len(evaluation_loss)))

    eval_loss = sum(evaluation_loss)/len(evaluation_loss)
    time_t1 = time.time()
    cost_time = format_time(time_t1 - time_t0)
    print("Evaluation loss: {} ; Data iteration{} cost time: {}".format(eval_loss, data_step, cost_time))


def main(args):
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        dist.barrier()
        ddp = True
    else:
        ddp = False

    batch_size = args.batch_size
    vocab_size = 5000

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    model = buildBERT(vocab_size)
    model.cuda(args.local_rank)
    if ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        gpu_num = torch.distributed.get_world_size()
    else:
        gpu_num = 1

    train_iter = search_path("attention", "./tokenized_data/train/")
    val_iter = search_path("attention", "./tokenized_data/val/")

    optimizer = AdamW(model.parameters(), lr=args.lr / 5, weight_decay=args.weight_decay)
    optimizer_con = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    for epoch in tqdm(range(args.epochs), desc="Training Epoch"):
        for step in tqdm(range(train_iter), desc="training iteration"):
            input_ids_train = torch.load("./tokenized_data/train/train_input_ids{}.pt".format(step))
            attention_masks_train = torch.load("./tokenized_data/train/train_attention_mask{}.pt".format(step))
            token_type_ids_train = torch.load("./tokenized_data/train/train_token_type_ids{}.pt".format(step))
            labels_train = torch.load("./tokenized_data/train/train_labels{}.pt".format(step))

            input_ids_train = torch.cat(input_ids_train, dim=0)
            attention_masks_train = torch.cat(attention_masks_train, dim=0)
            token_type_ids_train = torch.cat(token_type_ids_train, dim=0)
            labels_train = torch.cat(labels_train, dim=0)

            train_list = [input_ids_train, attention_masks_train, token_type_ids_train, labels_train]
            train_dataloader, train_sampler = generate_dataloader(train_list, "train", ddp, batch_size)

            config = Config()
            config.training_config(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay, device=device)

            train(model, train_dataloader, train_sampler, config, optimizer, optimizer_con, scaler, step)

        for step in tqdm(range(val_iter), desc="evaluation iteration"):
            input_ids_val = torch.load("./tokenized_data/val/val_input_ids{}.pt".format(step))
            attention_masks_val = torch.load("./tokenized_data/val/val_attention_mask{}.pt".format(step))
            token_type_ids_val = torch.load("./tokenized_data/val/val_token_type_ids{}.pt".format(step))
            labels_val = torch.load("./tokenized_data/val/val_labels{}.pt".format(step))

            input_ids_val = torch.cat(input_ids_val, dim=0)
            attention_masks_val = torch.cat(attention_masks_val, dim=0)
            token_type_ids_val = torch.cat(token_type_ids_val, dim=0)
            labels_val = torch.cat(labels_val, dim=0)

            val_list = [input_ids_val, attention_masks_val, token_type_ids_val, labels_val]
            val_dataloader, val_sampler = generate_dataloader(val_list, "val", ddp, batch_size)

            config = Config()
            config.training_config(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay, device=device)

            evaluate(model, val_dataloader, val_sampler, config, step)

    torch.save(model, 'bert_model/urlBERT.pt')


if __name__ == "__main__":

    seed_val = 2024
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    args = options.args_parser()
    main(args)
