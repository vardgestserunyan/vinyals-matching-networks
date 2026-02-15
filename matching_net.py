import torch
import random
import numpy as np
from torch import nn, optim, utils
from torchvision.datasets import Omniglot
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt
from tqdm import trange



def get_data():

    transform = Compose([Resize(size=(28,28)), 
                         ToTensor()])
    omni_back = Omniglot(root="./data", background=True, download=True, transform=transform)
    omni_eval = Omniglot(root="./data", background=False, download=True, transform=transform)

    back_dict = {}
    for idx, (_, label) in enumerate(omni_back):
        if not label in back_dict:
            back_dict[label] = []
        back_dict[label].append(idx)

    eval_dict = {}
    for idx, (_, label) in enumerate(omni_eval):
        if not label in eval_dict:
            eval_dict[label] = []
        eval_dict[label].append(idx)
        
    return omni_back, back_dict, omni_eval, eval_dict


def get_loader(omni_dset, dset_dict, num_ways, num_shots):

    num_char = len(omni_dset._characters)
    selected_classes = torch.randperm(num_char)[:num_ways] 

    supp_idx, query_idx = [], []
    for label in selected_classes:
        selected_indices = dset_dict[int(label)]
        random.shuffle(selected_indices)
        supp_idx.extend(selected_indices[:num_shots])
        query_idx.extend(selected_indices[num_shots:])
    

    supp_slice = utils.data.Subset(omni_dset, supp_idx)
    omni_supp_ldr = utils.data.DataLoader(supp_slice, batch_size=len(supp_slice), shuffle=True)

    query_slice = utils.data.Subset(omni_dset, query_idx)
    omni_query_ldr = utils.data.DataLoader(query_slice, batch_size=5, shuffle=True)

    return omni_supp_ldr, omni_query_ldr 


class MatchGRU(nn.Module):
    def __init__(self, hyper):
        super(MatchGRU, self).__init__()

        conv_emb_list = []
        channels_in = 1
        for _ in range(hyper["num_conv"]):
            conv_emb_list.extend([nn.Conv2d(channels_in, hyper["conv_size"], hyper["kernel_size"], \
                                            padding=(hyper["kernel_size"]-1)//2, padding_mode='replicate'), \
                                  nn.BatchNorm2d(hyper["conv_size"]), \
                                  nn.LeakyReLU(0.2), \
                                  nn.MaxPool2d(hyper["pool_size"])])
            channels_in =  hyper["conv_size"]
        conv_emb_list.extend([nn.Flatten()])

        self.conv_emb = nn.Sequential(*conv_emb_list)
        self.context_gru = nn.GRU(hyper["conv_size"], hyper["conv_size"], num_layers=1, bidirectional=True)
        self.cotext_norm = nn.LayerNorm(2*hyper["conv_size"])
        self.attn_gru = nn.GRU(hyper["conv_size"], 2*hyper["conv_size"], num_layers=1, batch_first=True)
        self.attn_norm = nn.LayerNorm(2*hyper["conv_size"])
        
        self.hyper = hyper
        self.device = hyper["device"]
        self.to(self.device)

    def forward(self, supp_x, supp_y_raw, query_x):

        query_size = query_x.shape[0]
        supp_y = nn.functional.one_hot(supp_y_raw).to(torch.float32)
        
        # Obtain initial embeddings
        supp_embed = self.conv_emb(supp_x)
        query_embed = self.conv_emb(query_x).unsqueeze(dim=1)

        # Obtain context-aware embeddings of the support
        context_gru_h, _ = self.context_gru(supp_embed) # supp_embed 50x128, context_gru_h 50x128
        context_gru_h = self.cotext_norm(context_gru_h)
        context_embed = context_gru_h[:,:hyper["conv_size"]] + context_gru_h[:,hyper["conv_size"]:] + supp_embed # context_embed 50x128

        # Obtain context-aware embeddings of the query
        k = self.hyper["num_unrolls"]
        attn_gru_h_prev = torch.zeros((1, query_size, hyper["conv_size"])).to(self.device)
        for idx in range(k):

            #inner_prod = torch.matmul(attn_gru_h_prev[:,:,:hyper["conv_size"]],context_embed.transpose(0,1))
            cos_dist = nn.functional.cosine_similarity( attn_gru_h_prev[:,:,:hyper["conv_size"]], context_embed.unsqueeze(1), dim=2)
            #weighing = nn.functional.softmax(inner_prod, dim=2)
            weighing = nn.functional.softmax(cos_dist.transpose(0,1).unsqueeze(0), dim=2)
            readout = torch.matmul(weighing,context_embed)

            attn_gru_h_in = attn_gru_h_prev[:,:,:hyper["conv_size"]] + query_embed.transpose(0,1)
            attn_gru_h_in = torch.concat([attn_gru_h_in,readout], dim=2)

            _, attn_gru_h_prev = self.attn_gru(query_embed, attn_gru_h_in)
            attn_gru_h_prev = self.attn_norm(attn_gru_h_prev)


        # Obtain attention-weighted labels
        cos_dist = nn.functional.cosine_similarity(context_embed.unsqueeze(1), attn_gru_h_prev[:,:,:hyper["conv_size"]], dim=2)
        weights = nn.functional.softmax( 10*cos_dist, dim=0 )
        eval_y_est = torch.matmul( weights.transpose(0,1), supp_y )
        
        return eval_y_est

    def train_episode(self, model_input_supp, model_input_query, optimizer):

        self.train()

        # Obtain image embeddings of the support and the query
        supp_x, supp_y = next(iter(model_input_supp))
        supp_x, supp_y = supp_x.to(self.device), supp_y.to(self.device)
            
        # Encode the labels
        label_list = supp_y.unique()
        supp_y = 1*torch.stack([ item == label_list for item in supp_y ], dim=0)
        supp_y = supp_y.argmax(dim=1)

        # Run batches of the query set
        sum_loss, acc, num_samples = 0, 0, 0
        for idx, query_batch in enumerate(model_input_query):

            # Obtain the query data
            query_x, query_y_tgt =  query_batch
            query_x, query_y_tgt = query_x.to(self.device), query_y_tgt.to(self.device)
            query_y_tgt = 1*torch.stack([ item == label_list for item in query_y_tgt ], dim=0)
            query_y_tgt = query_y_tgt.argmax(dim=1)

            # Pass through the model
            eval_y_est = self.forward(supp_x, supp_y, query_x)

            # Compute and propagate the loss
            eval_y_est_mod = torch.log(eval_y_est+1e-8)
            loss = nn.functional.nll_loss(eval_y_est_mod, query_y_tgt) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sum_loss += loss.item()

            # Compute the accuracy
            acc += sum(eval_y_est_mod.argmax(dim=1) == query_y_tgt)
            num_samples += len(eval_y_est_mod)

            #print(sum(eval_y_est_mod.argmax(dim=1) == query_y_tgt)/len(eval_y_est_mod))
        
        #print(sum_loss)

        return sum_loss/num_samples, acc/num_samples
    
    def test_eval(self, model_input_supp, model_input_query):

        self.eval()

        # Obtain image embeddings of the support and the query
        supp_x, supp_y = next(iter(model_input_supp))
        supp_x, supp_y = supp_x.to(self.device), supp_y.to(self.device)
            
        # Encode the labels
        label_list = supp_y.unique()
        supp_y = 1*torch.stack([ item == label_list for item in supp_y ], dim=0)
        supp_y = supp_y.argmax(dim=1)

        with torch.no_grad():

            # Run batches of the query set
            sum_loss, acc, num_samples = 0, 0, 0
            for idx, query_batch in enumerate(model_input_query):

                # Obtain the query data
                query_x, query_y_tgt =  query_batch
                query_x, query_y_tgt = query_x.to(self.device), query_y_tgt.to(self.device)
                query_y_tgt = 1*torch.stack([ item == label_list for item in query_y_tgt ], dim=0)
                query_y_tgt = query_y_tgt.argmax(dim=1)

                # Pass through the model
                eval_y_est = self.forward(supp_x, supp_y, query_x)

                # Compute and propagate the loss
                eval_y_est_mod = torch.log(eval_y_est+1e-8)
                loss = nn.functional.nll_loss(eval_y_est_mod, query_y_tgt)
                sum_loss += loss.item()

                # Compute the accuracy
                acc += sum(eval_y_est_mod.argmax(dim=1) == query_y_tgt)
                num_samples += len(eval_y_est_mod)

                #print(sum(eval_y_est_mod.argmax(dim=1) == query_y_tgt)/len(eval_y_est_mod))
                
        #print(sum_loss)
        return sum_loss/num_samples, acc/num_samples


    def train_test_multiepisode(self, optimizer, num_episodes, num_ways, num_shots):
        omni_back, back_dict, omni_eval, eval_dict = get_data()
        loss_hist_back, acc_hist_back = np.zeros(num_episodes), np.zeros(num_episodes)
        loss_hist_eval, acc_hist_eval = np.zeros(num_episodes), np.zeros(num_episodes)
        for idx in trange(num_episodes):
            model_input_supp, model_input_query = get_loader(omni_back, back_dict, num_ways, num_shots)
            model_input_supp, model_input_query = model_input_supp, model_input_query
            loss_back, acc_back = matchgru_model.train_episode( model_input_supp, model_input_query, optimizer)
            loss_hist_back[idx] = loss_back
            acc_hist_back[idx] = acc_back

            model_input_supp, model_input_query = get_loader(omni_eval, eval_dict, num_ways, num_shots)
            model_input_supp, model_input_query = model_input_supp, model_input_query
            loss_eval, acc_eval = matchgru_model.test_eval( model_input_supp, model_input_query)
            loss_hist_eval[idx] = loss_eval
            acc_hist_eval[idx] = acc_eval

        return loss_hist_back, loss_hist_eval, acc_hist_back, acc_hist_eval





# Obtain the data


# Instantiate the model and run training
device = "cuda" if torch.cuda.is_available() else ( "mps" if torch.backends.mps.is_available() else "cpu")
num_episodes = 350
num_ways, num_shots = 20, 5
hyper = {"conv_size": 64, "kernel_size": 3, "pool_size": 2, "num_conv": 4, \
         "device": device, "num_unrolls": 5}
matchgru_model = MatchGRU(hyper)
optimizer = optim.Adam(matchgru_model.parameters(), lr=1e-4)
loss_back, loss_eval, acc_back, acc_eval = matchgru_model.train_test_multiepisode(optimizer, num_episodes, num_ways, num_shots)


# Plot the loss
avg_window = 50
loss_back_avg, loss_eval_avg = np.cumsum(loss_back), np.cumsum(loss_eval)
loss_back_avg = (loss_back_avg[avg_window:]-loss_back_avg[:-avg_window])/avg_window
loss_eval_avg = (loss_eval_avg[avg_window:]-loss_eval_avg[:-avg_window])/avg_window

fig, ax = plt.subplots(figsize=(12,6), ncols=1, nrows=2)
ax[0].plot(loss_back, alpha=0.5, color="orange")
ax[0].plot(loss_eval, alpha=0.5, color="blue")

ax[0].plot(range(avg_window, len(loss_back)), loss_back_avg, color="orange")
ax[0].plot(range(avg_window, len(loss_eval)), loss_eval_avg, color="blue")

ax[0].set_xlabel("Episode")
ax[0].set_ylabel("NL Loss")
ax[0].set_title("Learning GRU-based Matching Nets")
ax[0].legend(["Background", "Evaluation"])

# Plot the accuracy
acc_back_avg, acc_eval_avg = np.cumsum(acc_back), np.cumsum(acc_eval)
acc_back_avg = (acc_back_avg[avg_window:]-acc_back_avg[:-avg_window])/avg_window
acc_eval_avg = (acc_eval_avg[avg_window:]-acc_eval_avg[:-avg_window])/avg_window

ax[1].plot(acc_back, alpha=0.5, color="orange")
ax[1].plot(acc_eval, alpha=0.5, color="blue")

ax[1].plot(range(avg_window, len(acc_back)), acc_back_avg, color="orange")
ax[1].plot(range(avg_window, len(acc_eval)), acc_eval_avg, color="blue")

ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Accuracy of GRU-based Matching Nets")
ax[1].legend(["Background", "Evaluation"])

plt.show()