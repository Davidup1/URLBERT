from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch


def generate_dataloader(data_list, type_dataloader: str, ddp, batch_size):
    dataset = TensorDataset(data_list[0], data_list[1], data_list[2], data_list[3])

    if ddp:
        if type_dataloader == "train":
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size
            )
            return dataloader, sampler
        elif type_dataloader == "val":
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size
            )
            return dataloader, sampler
        else:
            return None
    else:
        if type_dataloader == "train":
            dataloader = DataLoader(
                dataset,
                sampler=RandomSampler(dataset),
                batch_size=batch_size
            )
            return dataloader
        elif type_dataloader == "val":
            dataloader = DataLoader(
                dataset,
                sampler=SequentialSampler(dataset),
                batch_size=batch_size
            )
            return dataloader
        else:
            return None
