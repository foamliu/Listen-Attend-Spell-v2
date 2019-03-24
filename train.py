import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from config import device, grad_clip, print_freq, vocab_size
from data_gen import AiShellDataset, pad_collate
from models import Encoder, Decoder, Seq2Seq
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        encoder = Encoder(args.input_dim, args.encoder_hidden_size, args.num_layers)
        decoder = Decoder(vocab_size, args.embedding_dim, args.decoder_hidden_size)

        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                        lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                         lr=args.lr)

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

    # Custom dataloaders
    train_dataset = AiShellDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               shuffle=True)
    valid_dataset = AiShellDataset('dev')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               shuffle=False, drop_last=True)

    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.2)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        scheduler.step()

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)
        logger.info('[Training] Accuracy : {:.4f}'.format(train_loss))

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           encoder=encoder,
                           decoder=decoder)

        logger.info('[Validate] Accuracy : {:.4f}'.format(valid_loss))

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, optimizer, best_loss, is_best)


def train(train_loader, encoder, decoder, optimizer, epoch, logger):
    encoder.train()  # train mode (dropout and batchnorm is used)
    decoder.train()

    model = Seq2Seq(encoder, decoder)

    losses = AverageMeter()

    # Batches
    for i, (features, trns, input_lengths) in enumerate(train_loader):
        # Move to GPU, if available
        features = features.float().to(device)
        trns = trns.long().to(device)
        input_lengths = input_lengths.long().to(device)

        # Forward prop.
        loss = model(features, input_lengths, trns)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, encoder, decoder):
    encoder.eval()
    decoder.eval()

    model = Seq2Seq(encoder, decoder)

    losses = AverageMeter()

    # Batches
    for i, (features, trns, input_lengths) in enumerate(valid_loader):
        # Move to GPU, if available
        features = features.float().to(device)
        trns = trns.long().to(device)
        input_lengths = input_lengths.long().to(device)

        # Forward prop.
        loss = model(features, input_lengths, trns)

        # Keep track of metrics
        losses.update(loss.item())

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
