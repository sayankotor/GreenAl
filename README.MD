1) To start gpt-small training 

**python3 transformer_from_scratch_tt.py --rank 0**

1) To start gpt-small with decomposed c_fc, c_proj 

**python3 transformer_from_scratch_tt.py --rank desired_rank**


i.e. weights with torch.Size([1024, 768]) => [torch.Size([1, 16, 16, rank]), torch.Size([128, 16, 16, rank]), torch.Size([rank, 3, 12, 1])]

