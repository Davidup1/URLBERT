import torch
from torch import nn
import torch.nn.functional as F
import copy


def kl(inputs, targets, reduction="sum"):
    """
    kl_div
    inputs：tensor，logits
    targets：tensor，logits
    """
    loss = F.kl_div(F.log_softmax(inputs, dim=-1),
                    F.softmax(targets, dim=-1),
                    reduction=reduction)
    return loss


def adv_project(grad, norm_type='inf', eps=1e-6):
    """
    L0,L1,L2
    """
    if norm_type == 'l2':
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
    elif norm_type == 'l1':
        direction = grad.sign()
    else:
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
    return direction


# ContrastiveLoss Define
def calculateContrastiveLoss(emb_i, emb_j, batch_size, temperature, device):
    z_i = F.normalize(emb_i, dim=1)
    z_j = F.normalize(emb_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temperature)
    negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device)
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss = torch.sum(loss_partial) / (2 * batch_size)
    return loss


# Generate loss
def dropAlloss(model: nn.Module, input_ids, token_type, atten_masks, batch_size, device):
    presentation = []
    logits = []

    for i in range(2):
        output = model(input_ids,
                       token_type_ids=token_type,
                       attention_mask=atten_masks,
                       output_hidden_states=True,
                       )
        hidden_states = output.hidden_states[0]
        logits.append(output.logits)
        presentation.append(hidden_states)
    logits1 = logits[0]
    logits2 = logits[1]
    presentation1 = presentation[0]
    presentation2 = presentation[1]

    noise1 = presentation1.data.new(presentation1.size()).normal_(0, 1) * 1e-5
    noise1.requires_grad_()
    noise2 = presentation2.data.new(presentation2.size()).normal_(0, 1) * 1e-5
    noise2.requires_grad_()
    epochs = 2
    for i in range(epochs):
        new_presentation1 = presentation1.data.detach() + noise1
        new_presentation2 = presentation2.data.detach() + noise2
        adv_output1 = model(inputs_embeds=new_presentation1,
                            token_type_ids=token_type,
                            attention_mask=atten_masks,
                            )
        adv_output2 = model(inputs_embeds=new_presentation2,
                            token_type_ids=token_type,
                            attention_mask=atten_masks,
                            )
        adv_logits1 = adv_output1.logits
        adv_logits2 = adv_output2.logits
        adv_loss1 = kl(adv_logits1, logits1.detach(), reduction="batchmean")
        adv_loss2 = kl(adv_logits2, logits2.detach(), reduction="batchmean")
        delta_grad1, = torch.autograd.grad(adv_loss1, noise1, only_inputs=True)
        delta_grad2, = torch.autograd.grad(adv_loss2, noise2, only_inputs=True)
        norm1 = delta_grad1.norm()
        norm2 = delta_grad2.norm()


        if torch.isnan(norm1) or torch.isinf(norm1) or torch.isnan(norm2) or torch.isinf(norm2):
            return None

        # line 6 inner sum
        noise1 = noise1 + delta_grad1 * 1e-1
        noise2 = noise2 + delta_grad2 * 1e-5
        # line 6 projection
        noise1 = adv_project(noise1, norm_type='l2', eps=1e-6)
        noise2 = adv_project(noise2, norm_type="l2", eps=1e-6)
        new_presentation1 = presentation1.data.detach() + noise1
        new_presentation1 = new_presentation1.detach()
        new_presentation2 = presentation2.data.detach() + noise2
        new_presentation2 = new_presentation2.detach()
    adv_output1 = model(inputs_embeds=new_presentation1,
                        token_type_ids=token_type,
                        attention_mask=atten_masks,
                        output_hidden_states=True,
                        )
    adv_output2 = model(inputs_embeds=new_presentation2,
                        token_type_ids=token_type,
                        attention_mask=atten_masks,
                        output_hidden_states=True,
                        )
    adv_hidden_states1 = adv_output1.hidden_states[-1][:, 0, :]
    adv_hidden_states2 = adv_output2.hidden_states[-1][:, 0, :]
    batch_size = len(input_ids)
    loss = calculateContrastiveLoss(adv_hidden_states1, adv_hidden_states2, batch_size, 0.1, device)
    return loss
