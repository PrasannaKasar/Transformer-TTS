from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from utils import *

device = 'cuda'

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(model, test_loader, writer, epoch):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # pbar.set_description("Processing at epoch %d"%epoch)
            character, mel, mel_input, pos_text, pos_mel, _ = data
            pos_mel.to(device)
            stop_tokens = torch.abs(pos_mel.ne(0).type(t.float) - 1).to(device)
            # print(mel[0].shape)
            # print(mel[0][len(mel[0])-1])
            character = character.to(device)
            mel = mel.to(device)
            mel_input = mel_input.to(device)
            pos_text = pos_text.to(device)
            pos_mel = pos_mel.to(device)
            # print(stop_tokens[0])
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = model.forward(character, mel_input, pos_text, pos_mel)
            # print(type(attn_probs))   # Check if it's a list
            # print(len(attn_probs))    # Check how many elements are in the list
            # print(attn_probs[0].shape)  # Print the shape of the first attention matrix
            attn_matrix = attn_probs[-1]
            input_lengths = torch.sum(character != 0, dim=1).to(device)
            output_lengths = torch.sum(mel != 0, dim=1).to(device)
            output_lengths = output_lengths[:, 0]
            # print(input_lengths.shape)
            # print(output_lengths.shape)
            stop_preds = stop_preds.squeeze(-1)

            # Assuming attention_matrices is of shape (4 * batch_size, max_output_len, max_input_len)
            # print(attn_matrix.shape)
            attn_matrix = attn_matrix.view(hp.batch_size, 4, max(output_lengths), max(input_lengths))
            # You can then average over the heads or use another aggregation method
            attn_matrix = attn_matrix.mean(dim=1)  # Shape: (batch_size, max_output_len, max_input_len)


            total_attn_loss = 0.0
    
            # Guided attention: Compute attention loss for each example in the batch
            for b in range(attn_matrix.size(0)):  # Iterate over batch size
                # Get the true input and output lengths for this batch item
                N = input_lengths[b].item()  # Input length (characters or phonemes)
                T = output_lengths[b].item()  # Output length (mel frames)
                
                # Compute the guided attention weight matrix for this example
                W = guided_attention(N, T, g=0.2)  # Use the function you found
                
                # Convert W to a torch tensor and move it to the appropriate device
                W = torch.tensor(W).to(attn_matrix.device)  # Shape: [T, N]
                
                # Slice the attention matrix for the valid part (before padding)
                attn_slice = attn_matrix[b, :T, :N]  # Shape: [T, N]
                # attn_slice = attn_slice.T
                W = W.T
                # print("W shape:", W.shape)
                # print("attn_slice shape:", attn_slice.shape)

                
                # Compute the attention loss for this batch item
                attn_loss = torch.mean(W * attn_slice)
                
                # Accumulate the total attention loss across the batch
                total_attn_loss += attn_loss
            
            # Average the total attention loss across the batch
            total_attn_loss = total_attn_loss / attn_matrix.size(0)  # Normalize by batch size
            
            mel_loss = nn.MSELoss()(mel_pred, mel)
            post_mel_loss = nn.MSELoss()(postnet_pred, mel)
            criterion = nn.BCEWithLogitsLoss()
            # attn_loss = guided_attention(attn_matrix, input_lengths, output_lengths)
            # stop_token_loss = criterion(stop_preds, stop_tokens) * 8.0
            lamda = 0.2
            loss = mel_loss + post_mel_loss + lamda * total_attn_loss 
            # + stop_token_loss
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    writer.add_scalars('test_loss_per_epoch',{
                    'loss':avg_test_loss
                }, epoch)
    print(f"Test Loss: {avg_test_loss}")    

def main():

    dataset = get_dataset()
    global_step = 0

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    m = nn.DataParallel(Model().to(device))

    m.train()
    optimizer = torch.optim.Adam(m.parameters(), lr=hp.lr)

    pos_weight = torch.FloatTensor([5.]).to(device)
    writer = SummaryWriter()
    
    for epoch in range(hp.epochs):

        train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=collate_fn_transformer, drop_last=True, num_workers=4)
        pbar = tqdm(train_loader)
        epoch_loss = 0
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)
                
            character, mel, mel_input, pos_text, pos_mel, _ = data
            pos_mel.to(device)
            stop_tokens = torch.abs(pos_mel.ne(0).type(t.float) - 1).to(device)
            # print(mel[0].shape)
            # print(mel[0][len(mel[0])-1])
            character = character.to(device)
            mel = mel.to(device)
            mel_input = mel_input.to(device)
            pos_text = pos_text.to(device)
            pos_mel = pos_mel.to(device)
            # print(stop_tokens[0])
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)
            # print(type(attn_probs))   # Check if it's a list
            # print(len(attn_probs))    # Check how many elements are in the list
            # print(attn_probs[0].shape)  # Print the shape of the first attention matrix
            attn_matrix = attn_probs[-1]
            input_lengths = torch.sum(character != 0, dim=1).to(device)
            output_lengths = torch.sum(mel != 0, dim=1).to(device)
            output_lengths = output_lengths[:, 0]
            # print(input_lengths.shape)
            # print(output_lengths.shape)
            stop_preds = stop_preds.squeeze(-1)

            # Assuming attention_matrices is of shape (4 * batch_size, max_output_len, max_input_len)
            attn_matrix = attn_matrix.view(hp.batch_size, 4, max(output_lengths), max(input_lengths))
            # You can then average over the heads or use another aggregation method
            attn_matrix = attn_matrix.mean(dim=1)  # Shape: (batch_size, max_output_len, max_input_len)

            total_attn_loss = 0.0
    
            # Guided attention: Compute attention loss for each example in the batch
            for b in range(attn_matrix.size(0)):  # Iterate over batch size
                # Get the true input and output lengths for this batch item
                N = input_lengths[b].item()  # Input length (characters or phonemes)
                T = output_lengths[b].item()  # Output length (mel frames)
                
                # Compute the guided attention weight matrix for this example
                W = guided_attention(N, T, g=0.2)  # Use the function you found
                
                # Convert W to a torch tensor and move it to the appropriate device
                W = torch.tensor(W).to(attn_matrix.device)  # Shape: [T, N]
                
                # Slice the attention matrix for the valid part (before padding)
                attn_slice = attn_matrix[b, :T, :N]  # Shape: [T, N]
                # attn_slice = attn_slice.T
                W = W.T
                # print("W shape:", W.shape)
                # print("attn_slice shape:", attn_slice.shape)

                
                # Compute the attention loss for this batch item
                attn_loss = torch.mean(W * attn_slice)
                
                # Accumulate the total attention loss across the batch
                total_attn_loss += attn_loss
            
            # Average the total attention loss across the batch
            total_attn_loss = total_attn_loss / attn_matrix.size(0)  # Normalize by batch size
            
            mel_loss = nn.MSELoss()(mel_pred, mel)
            post_mel_loss = nn.MSELoss()(postnet_pred, mel)
            criterion = nn.BCEWithLogitsLoss()
            # attn_loss = guided_attention(attn_matrix, input_lengths, output_lengths)
            # stop_token_loss = criterion(stop_preds, stop_tokens) * 8.0
            lamda = 0.2
            loss = mel_loss + post_mel_loss + lamda * total_attn_loss
            # + stop_token_loss
            epoch_loss += loss.item()
            writer.add_scalars('training_loss',{
                    'mel_loss':mel_loss,
                    'post_mel_loss':post_mel_loss,
                    'total_attn_loss':total_attn_loss
                    # 'stop_token_loss': stop_token_loss
                }, global_step)
            writer.add_scalars('alphas',{
                    'encoder_alpha':m.module.encoder.alpha.data,
                    'decoder_alpha':m.module.decoder.alpha.data,
                }, global_step)
            
            
            if global_step % hp.image_step == 1:
                
                for i, prob in enumerate(attn_probs):
                    
                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_%d_0'%global_step, x, i*4+j)
                
                for i, prob in enumerate(attns_enc):
                    num_h = prob.size(0)
                    
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)
            
                for i, prob in enumerate(attns_dec):

                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)
                
            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            
            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                torch.save({'model':m.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))
                
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalars('training_loss_per_epoch',{
                    'loss':avg_loss,
                }, epoch)
        print(f"Loss at epoch {epoch} = {avg_loss}")

        test(m, test_loader, writer, epoch)


if __name__ == '__main__':
    main()
