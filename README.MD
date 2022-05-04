1) To start gpt-small training 

**python3 transformer_from_scratch_tt.py --rank 0**

1) To start gpt-small with decomposed c_fc, c_proj 

**python3 transformer_from_scratch_tt.py --rank desired_rank**


i.e. weights with torch.Size([1024, 768]) => [torch.Size([1, 16, 16, rank]), torch.Size([128, 16, 16, rank]), torch.Size([rank, 3, 12, 1])]


optimizer = AdamW(parameters, lr=6.25e-5, eps=1e-8)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=4000, num_training_steps = total_steps)
