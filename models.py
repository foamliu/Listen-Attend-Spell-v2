import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import PAD_token, SOS_token, EOS_token
from utils import pad_list


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        loss = self.decoder(padded_target, encoder_padded_outputs)
        return loss.mean()

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs, _ = self.encoder(input, input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=True):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, input_x, enc_len):
        total_length = input_x.size(1)  # get the max sequence length
        # print('total_length: ' + str(total_length))
        # print('input_x.size(): ' + str(input_x.size()))
        packed_input = pack_padded_sequence(input_x, enc_len, batch_first=True)
        # print('enc_len: ' + str(enc_len))
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        # Hyper parameters
        # embedding + output
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # rnn
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_hidden_size = hidden_size  # must be equal now
        # Components
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(self.embedding_dim +
                                 self.encoder_hidden_size, self.hidden_size)]
        for l in range(1, self.num_layers):
            self.rnn += [nn.LSTMCell(self.hidden_size, self.hidden_size)]
        self.attention = DotProductAttention()
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_hidden_size + self.hidden_size,
                      self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.vocab_size))

    def zero_state(self, encoder_padded_outputs, H=None):
        N = encoder_padded_outputs.size(0)
        H = self.hidden_size if H == None else H
        return encoder_padded_outputs.new_zeros(N, H)

    def forward(self, padded_input, encoder_padded_outputs):
        """
        Args:
            padded_input: N x To
            # encoder_hidden: (num_layers * num_directions) x N x H
            encoder_padded_outputs: N x Ti x H
        Returns:
        """
        # *********Get Input and Output
        # from espnet/Decoder.forward()
        ys = [y[y != PAD_token] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([EOS_token])
        sos = ys[0].new([SOS_token])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, EOS_token)
        ys_out_pad = pad_list(ys_out, PAD_token)
        assert ys_in_pad.size() == ys_out_pad.size()
        batch_size = ys_in_pad.size(0)
        output_length = ys_in_pad.size(1)

        # *********Init decoder rnn
        h_list = [self.zero_state(encoder_padded_outputs)]
        c_list = [self.zero_state(encoder_padded_outputs)]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_padded_outputs))
            c_list.append(self.zero_state(encoder_padded_outputs))
        att_c = self.zero_state(encoder_padded_outputs,
                                H=encoder_padded_outputs.size(2))
        y_all = []

        # **********LAS: 1. decoder rnn 2. attention 3. concate and MLP
        embedded = self.embedding(ys_in_pad)
        for t in range(output_length):
            # step 1. decoder RNN: s_i = RNN(s_i−1,y_i−1,c_i−1)
            rnn_input = torch.cat((embedded[:, t, :], att_c), dim=1)
            h_list[0], c_list[0] = self.rnn[0](
                rnn_input, (h_list[0], c_list[0]))
            for l in range(1, self.num_layers):
                h_list[l], c_list[l] = self.rnn[l](
                    h_list[l - 1], (h_list[l], c_list[l]))
            rnn_output = h_list[-1]  # below unsqueeze: (N x H) -> (N x 1 x H)
            # step 2. attention: c_i = AttentionContext(s_i,h)
            att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                                          encoder_padded_outputs)
            att_c = att_c.squeeze(dim=1)
            # step 3. concate s_i and c_i, and input to MLP
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            predicted_y_t = self.mlp(mlp_input)
            y_all.append(predicted_y_t)

        y_all = torch.stack(y_all, dim=1)  # N x To x C

        # **********Cross Entropy Loss
        # F.cross_entropy = NLL(log_softmax(input), target))
        y_all = y_all.view(batch_size * output_length, self.vocab_size)
        ce_loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                  ignore_index=PAD_token,
                                  reduction='mean')

        return ce_loss

    def recognize_beam(self, encoder_outputs, char_list, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam
        Returns:
            nbest_hyps:
        """
        # search params
        beam = args.beam_size
        nbest = args.nbest
        if args.decode_max_len == 0:
            maxlen = encoder_outputs.size(0)
        else:
            maxlen = args.decode_max_len

        # *********Init decoder rnn
        h_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        c_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
            c_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
        att_c = self.zero_state(encoder_outputs.unsqueeze(0),
                                H=encoder_outputs.unsqueeze(0).size(2))
        # prepare sos
        y = SOS_token
        vy = encoder_outputs.new_zeros(1).long()

        hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'h_prev': h_list,
               'a_prev': att_c}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                # vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                embedded = self.embedding(vy)
                # embedded.unsqueeze(0)

                # step 1. decoder RNN: s_i = RNN(s_i−1,y_i−1,c_i−1)
                rnn_input = torch.cat((embedded, hyp['a_prev']), dim=1)
                h_list[0], c_list[0] = self.rnn[0](
                    rnn_input, (hyp['h_prev'][0], hyp['c_prev'][0]))
                for l in range(1, self.num_layers):
                    h_list[l], c_list[l] = self.rnn[l](
                        h_list[l - 1], (hyp['h_prev'][l], hyp['c_prev'][l]))
                rnn_output = h_list[-1]

                # step 2. attention: c_i = AttentionContext(s_i,h)
                # below unsqueeze: (N x H) -> (N x 1 x H)
                att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                                              encoder_outputs.unsqueeze(0))
                att_c = att_c.squeeze(dim=1)

                # step 3. concate s_i and c_i, and input to MLP
                mlp_input = torch.cat((rnn_output, att_c), dim=1)
                predicted_y_t = self.mlp(mlp_input)
                local_scores = F.log_softmax(predicted_y_t, dim=1)
                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['h_prev'] = h_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_c[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(
                        local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:beam]
            # end for hyp in hyps
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'].append(EOS_token)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == EOS_token:
                    # hyp['score'] += (i + 1) * penalty
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            if len(hyps) > 0:
                pass
                # print('remeined hypothes: ' + str(len(hyps)))
            else:
                # print('no hypothesis. Finish decoding.')
                break

            # for hyp in hyps:
            #     print('hypo: ' + ''.join([char_list[int(x)]
            #                               for x in hyp['yseq'][1:]]))
        # end for i in range(maxlen)
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
                     :min(len(ended_hyps), nbest)]
        return nbest_hyps


class DotProductAttention(nn.Module):
    r"""Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.
    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, values):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H
        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        batch_size = queries.size(0)
        input_lengths = values.size(1)
        # (N, To, H) * (N, H, Ti) -> (N, To, Ti)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        attention_distribution = F.softmax(
            attention_scores.view(-1, input_lengths), dim=1).view(batch_size, -1, input_lengths)
        # (N, To, Ti) * (N, Ti, H) -> (N, To, H)
        attention_output = torch.bmm(attention_distribution, values)

        return attention_output, attention_distribution


if __name__ == "__main__":
    pass
