"""
@Author : Fendi Zhang <fendizh001@gmail.com>
@Start-Date : 2022-11-15
@Filename : function.py'
@Framework : Pytorch
"""

import torch
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

SLOT_PAD = 0


# Gaussian KL loss
def kl_loss(mu_i, sigma_i, mu_j, sigma_j, embed_dimension):
    '''
    #高斯分布、泊松分布、韦伯分布等
    Calculates KL-divergence between two DIAGONAL Gaussians.
    Reference: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians.
    '''
    sigma_ratio = sigma_j / sigma_i
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=1)
    mu_diff_sq = torch.sum((mu_i - mu_j) ** 2 / sigma_i, axis=1)
    ij_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    sigma_ratio = sigma_i / sigma_j
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=1)
    mu_diff_sq = torch.sum((mu_j - mu_i) ** 2 / sigma_j, axis=1)
    ji_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    kl_d = 0.5 * (ij_kl + ji_kl)
    return kl_d


def pad_label(lengths=None, labels=None):
    y = labels
    bsz = len(lengths)
    max_len = torch.max(lengths)
    padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
    for i in range(bsz):
        length = lengths[i]
        y_i = y[i]
        padded_y[i, 0:length] = torch.LongTensor(y_i)
    padded_y = padded_y.to(device)
    return padded_y


def nt_xent(loss, num, denom, temperature=1):
    loss = torch.exp(loss / temperature)
    cnts = torch.sum(num, dim=1)
    loss_num = torch.sum(loss * num, dim=1)
    loss_denom = torch.sum(loss * denom, dim=1)

    # sanity check
    nonzero_indexes = torch.where(cnts > 0)
    loss_num, loss_denom, cnts = loss_num[nonzero_indexes], loss_denom[nonzero_indexes], cnts[nonzero_indexes]
    loss_num, loss_denom, cnts = loss_num.to(device), loss_denom.to(device), cnts.to(device)

    loss_final = -torch.log2(loss_num) + torch.log2(loss_denom) + torch.log2(cnts)
    return loss_final


def euclidean_distance(a, b, normalize=False):
    if normalize:
        a = F.normalize(a)
        b = F.normalize(b)
    logits = ((a - b) ** 2).sum(dim=1)
    return logits


def remove_irrelevant_tokens_for_loss(self, original_embedding_mu, original_embedding_sigma, attention_mask, labels, model_args):
    attention_mask = attention_mask.to(device)
    active_indices = attention_mask.view(-1)
    active_indices = torch.where(active_indices == True)[0]

    output_embedding_mu = original_embedding_mu.view(-1, model_args.embedding_dim)[
        active_indices]  # args.emb_dim = 300 => 600
    output_embedding_sigma = original_embedding_sigma.view(-1, model_args.embedding_dim)[
        active_indices]  # args.emb_dim = 300 => 600
    # labels_straightened = padded_labels.view(-1)[active_indices]
    labels_straightened = labels.view(-1)[active_indices]
    # print("labels_straightened: {}".format(labels_straightened))

    # nonneg_indices = torch.where(labels_straightened >= 0)[0]
    # output_embedding_mu = output_embedding_mu[nonneg_indices]
    # output_embedding_sigma = output_embedding_sigma[nonneg_indices]
    # labels_straightened = labels_straightened[nonneg_indices]

    return output_embedding_mu, output_embedding_sigma, labels_straightened


def calculate_KL_or_euclidean(self, original_embedding_mu, original_embedding_sigma, attention_mask, labels,
                              nelgect_mutual_O=False, loss_type=None, model_args=None):
    # The strategy that creates embedding pairs in following manner
    # filtered_embedding | embedding ||| filtered_labels | labels
    # repeat_interleave |            ||| repeat_interleave |
    #                   | repeat     |||                   | repeat
    # extract only active parts that does not contain any paddings

    output_embedding_mu, output_embedding_sigma, labels_straightened = remove_irrelevant_tokens_for_loss(self,
                                                                                                         original_embedding_mu,
                                                                                                         original_embedding_sigma,
                                                                                                         attention_mask,
                                                                                                         labels,
                                                                                                         model_args,
                                                                                                         )

    # remove indices with zero labels, that is "O" classes
    if nelgect_mutual_O:
        filter_indices = torch.where(labels_straightened > 0)[0]
        filtered_embedding_mu = output_embedding_mu[filter_indices]
        filtered_embedding_sigma = output_embedding_sigma[filter_indices]
        filtered_labels = labels_straightened[filter_indices]
    else:
        filtered_embedding_mu = output_embedding_mu
        filtered_embedding_sigma = output_embedding_sigma
        filtered_labels = labels_straightened

    filtered_instances_nos = len(filtered_labels)

    # repeat interleave
    filtered_embedding_mu = torch.repeat_interleave(filtered_embedding_mu, len(output_embedding_mu), dim=0)
    filtered_embedding_sigma = torch.repeat_interleave(filtered_embedding_sigma, len(output_embedding_sigma), dim=0)
    filtered_labels = torch.repeat_interleave(filtered_labels, len(output_embedding_mu), dim=0)

    # only repeat
    repeated_output_embeddings_mu = output_embedding_mu.repeat(filtered_instances_nos, 1)
    repeated_output_embeddings_sigma = output_embedding_sigma.repeat(filtered_instances_nos, 1)
    repeated_labels = labels_straightened.repeat(filtered_instances_nos)

    # avoid losses with own self
    loss_mask = torch.all(filtered_embedding_mu != repeated_output_embeddings_mu, dim=-1).int()
    loss_weights = (filtered_labels == repeated_labels).int()
    loss_weights = loss_weights * loss_mask
    loss_weights = loss_weights.float()  # Fendi add on June 13rd, 2022

    # torch.LongTensor
    # loss_weights = torch.LongTensor(loss_weights) #Fendi

    # ensure that the vector sizes are of filtered_instances_nos * filtered_instances_nos
    # print("repeated_labels:{},len(repeated_labels):{}".format(repeated_labels,len(repeated_labels)))
    # print("filtered_instances_nos:{}".format(filtered_instances_nos))
    # print("filtered_instances_nos * filtered_instances_nos:{}".format(filtered_instances_nos * filtered_instances_nos))

    assert len(repeated_labels) == (
            filtered_instances_nos * filtered_instances_nos)  # "dimension is not of square shape."

    if loss_type == "euclidean":
        loss = -euclidean_distance(filtered_embedding_mu, repeated_output_embeddings_mu, normalize=True)

    elif loss_type == "KL":  # KL_divergence
        loss = -kl_loss(filtered_embedding_mu, filtered_embedding_sigma,
                        repeated_output_embeddings_mu, repeated_output_embeddings_sigma,
                        embed_dimension=model_args.embedding_dim)

    else:
        raise Exception("unknown loss type")

    # reshape the loss, loss_weight, and loss_mask
    loss = loss.view(filtered_instances_nos, filtered_instances_nos)
    loss_mask = loss_mask.view(filtered_instances_nos, filtered_instances_nos)
    loss_weights = loss_weights.view(filtered_instances_nos, filtered_instances_nos)

    loss_final = nt_xent(loss, loss_weights, loss_mask, temperature=1)
    # print("avg_gaussian_loss_final: ", torch.mean(loss_final))
    return torch.mean(loss_final)
