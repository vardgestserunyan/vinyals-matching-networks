This is a reproduction of the Vinyals et al.'s 2017 paper on one-shot learning https://arxiv.org/pdf/1606.04080. 
My reproduction focuses on the Omniglot portion of the paper (Section 4.1.1). 
I made two changes from the original. 
One, I replaced LSTMs in the original with GRUs. The main reason is that this work was done on my personal laptop and I wanted to avoid heavy computational requirements.
Two, I added layer normalization after GRU layers, which seems to have sped up the training noticeably. 
For 20-way 5-shot learning, my model showed around 95% accuracy, in line with paper's reported 98.5%. 
The difference of 3.5% percentage points is likely attributable to my use of GRUs as opposed to LSTMs.
