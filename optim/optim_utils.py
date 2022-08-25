import torch


def lr_scheduler(optimizer, num_epochs=200, decay_start_epoch=100, offset=0):
    """
        This function returns learning rate scheduler
    """
    def lmabda_lr_func(epoch):
        return 1 - max(0, epoch + offset - decay_start_epoch)/(num_epochs - decay_start_epoch)

    _lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lmabda_lr_func)

    return _lr_scheduler
