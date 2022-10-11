lr_epoch = 0
learning_rate_dict = {}

for i in range(8):
    learning_rate = 0.001 * 10 ** (-i / 7)
    learning_rate_dict[lr_epoch] = learning_rate
    lr_epoch += 3 ** i


def lr_scheduler(epoch, lr):
    try:
        lr = learning_rate_dict[epoch-1]
    except:
        lr = lr

    return lr
