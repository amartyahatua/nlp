def testGan(encoder, decoder, sentence, max_length=200):
    with torch.no_grad():
        #input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = sentence.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_outputs[0: input_length,:] = sentence
        # for ei in range(input_length):
        #     encoder_output, encoder_hidden = encoder(sentence[ei],
        #                                              encoder_hidden)
        #     encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]












# # create new sent
# logits = .squeeze()
# seq = logits.argmax(dim=0)
# print(ds.decode(seq))

def evaluateGAN(encoder, decoder, n=200):
    for i in range(n):
        generator = Generator(n_layers, block_dim)
        generator.eval()
        generator.load_state_dict(torch.load('generator.th', map_location='cpu'))

        noise = torch.from_numpy(np.random.normal(0, 20, (5, latent_dim))).float()
        z = generator(noise)

        pair = random.choice(pairs)
        #print('>', pair[0])
        #print('=', pair[1])
        output_words, attentions = testGan(encoder, decoder, z)
        #output_words, attentions = testGan(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')







encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
encoder1.load_state_dict(torch.load('encoder1.dict'))



attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
attn_decoder1.load_state_dict(torch.load('attn_decoder1.dict'))



evaluateGAN(encoder1,attn_decoder1)