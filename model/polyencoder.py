import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import MultiHeadCrossAttention
from transformers import BertModel, BertTokenizer

class PolyEncoder(nn.Module):
    def __init__(self, tokenizer=None, poly_m=10):
        super(PolyEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.resize_token_embeddings(len(tokenizer)) # resize token embeddings
        self.number_of_heads = 8
        self.MultiHeadCrossAttention = MultiHeadCrossAttention(768, self.number_of_heads)
        self.poly_m = poly_m
        self.poly_code_embedding = nn.Embedding(self.poly_m, 768)

        torch.nn.init.xavier_uniform_(self.poly_code_embedding.weight,768**(-0.5)) # xavier_uniform initialization

    def dot_attention(self, q, k, v):
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        attention_weights = F.softmax(attention_weights, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output
    
    def forward(self, inputs):
        context_ids, context_masks, gloss_ids, gloss_masks, word_ids = inputs["context_ids"], inputs["context_masks"], inputs["gloss_ids"], inputs["gloss_masks"], inputs["word_ids"]
        gloss_ids = gloss_ids.unsqueeze(1)

        batch_size, _, sequence_length = gloss_ids.shape
        # batch_size: 是一个batch的大小
        # _: 指的是一个句子有多少sequence
        # sequence_length: 指的是句子中token的数量
        gloss_ids = gloss_ids.squeeze(1)

        context_output = self.bert(context_ids, attention_mask=context_masks)[0]
        gloss_output = self.bert(gloss_ids, attention_mask=gloss_masks)[0][:, 0, :]
        word_output = self.bert(word_ids, attention_mask=gloss_masks)[0][:, 1, :]

        poly_codes_ids = torch.arange(self.poly_m,dtype=torch.long).to(context_ids.device)
        poly_codes_ids = poly_codes_ids.expand(batch_size, self.poly_m) # (batch_size, poly_m)
        poly_codes_embedding = self.poly_code_embedding(poly_codes_ids) # (batch_size, poly_m, 768)

        context_poly_embedding = self.dot_attention(poly_codes_embedding,context_output,context_output)
        gloss_poly_embedding = gloss_output.expand(self.poly_m, batch_size, sequence_length).transpose(0, 1)
        word_poly_embedding = word_output.expand(self.poly_m, batch_size, sequence_length).transpose(0, 1)

        gloss_poly_embedding = self.MultiHeadCrossAttention(word_poly_embedding,gloss_poly_embedding,gloss_poly_embedding)
        context_poly_embedding = self.MultiHeadCrossAttention(word_poly_embedding,context_poly_embedding,context_poly_embedding)
        
        context_embedding = self.dot_attention(gloss_poly_embedding, context_poly_embedding, context_poly_embedding).transpose(0, 1)
        gloss_embedding = gloss_poly_embedding.permute(1,2,0)
        representation = torch.matmul(context_embedding,gloss_embedding).sum(0)

        loss_mask = torch.eye(batch_size).to(context_ids.device)
        loss = F.log_softmax(representation, dim=1) * loss_mask
        loss = (-loss.sum(dim=1)).mean()
        return_dictionary = {"loss": loss,
                             "representation": representation}
        return return_dictionary
