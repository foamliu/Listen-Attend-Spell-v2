import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import StepLR

from config import device, grad_clip, print_freq
from data_gen import Thchs30Dataset
from models import Encoder, Decoder
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, accuracy, get_logger


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = 0
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        encoder = Encoder()
        decoder = Decoder()

        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                        lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                         lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = Thchs30Dataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = Thchs30Dataset('train')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        scheduler.step()

        # One epoch's training
        train_loss, train_top5_accs = train(train_loader=train_loader,
                                            encoder=encoder,
                                            decoder=decoder,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            logger=logger)
        # train_dataset.shuffle()
        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Train_Top5_Accuracy', train_top5_accs, epoch)

        # One epoch's validation
        valid_loss, valid_top5_accs = valid(valid_loader=valid_loader,
                                            encoder=encoder,
                                            decoder=decoder,
                                            epoch=epoch,
                                            logger=logger)

        # Check if there was an improvement
        is_best = valid_top5_accs > best_acc
        best_acc = max(valid_top5_accs, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, optimizer, best_acc, is_best)


def train(train_loader, encoder, decoder, criterion, optimizer, epoch, logger):
    encoder.train()  # train mode (dropout and batchnorm is used)
    decoder.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (feature, trn) in enumerate(train_loader):
        # Move to GPU, if available
        feature = feature.to(device)
        trn = trn.to(device)  # [N, 1]

        # Forward prop.
        embedding = encoder(feature)  # embedding => [N, 512]
        output = decoder(embedding, trn)  # class_id_out => [N, 10575]

        # Calculate loss
        loss = criterion(output, trn)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(output, trn, 5)
        top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))

    return losses.avg, top5_accs.avg


def valid(valid_loader, encoder, decoder, epoch, logger):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (feature, trn) in enumerate(valid_loader):
        # Move to GPU, if available
        feature = feature.to(device)
        trn = trn.to(device)  # [N, 1]

        # Forward prop.
        embedding = encoder(feature)  # embedding => [N, 512]
        output = decoder(embedding, trn)  # class_id_out => [N, 10575]

        loss = 0

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(output, trn, 5)
        top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(valid_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))

    return losses.avg, top5_accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
