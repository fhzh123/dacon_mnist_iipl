import torch.nn.functional as F

def loss_ce(output, label):
    loss = F.cross_entropy(output, label)
    return loss

def loss_kd(t_output, s_output, label, alpha, T):
    t_output = F.softmax(t_output/T, dim=1)
    s_output = F.log_softmax(s_output/T, dim=1)
    
    criterion = F.cross_entropy(s_output, label)
    loss_kl = F.kl_div(s_output, t_output, reduction='batchmean')
    loss =  loss_kl * (alpha * T * T) + criterion * (1. - alpha)
    return loss