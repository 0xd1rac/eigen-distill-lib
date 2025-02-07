import torch.nn.functional as F
import torch 

def KD_loss(student_log_probs, teacher_probs):
    """
    Kullback-Leibler divergence loss for knowledge distillation.
    Note: student_log_probs should be log-softmax outputs,
    teacher_probs should be softmax outputs.
    """
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
