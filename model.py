import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
         
        # We use bacth norm. in resnet101, so i use only last step for optimization batchnorm
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3):
        super(DecoderRNN, self).__init__()
        
        # We use .self for variable because we can access the attributes and methods of the class 
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        #torch.nn.Embedding(num_embeddin gs, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, 
        #_weight=None, device=None, dtype=None)

        #num_embeddings (int) – size of the dictionary of embeddings
        #embedding_dim (int) – the size of each embedding vector

        self.embedded = nn.Embedding(vocab_size, embed_size)
        self.LSTM = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        
    # I examine answer of the mentor for forward part
     # https://knowledge.udacity.com/questions/257564   
    def forward(self, features, captions):
        
        captions = captions[:,:-1] # discard <end>
        captions = self.embedded(captions[:,:-1])
        
        embed = torch.cat((features.unsqueeze(1), captions_embed), dim=1)
        
        prediction, (h, c)  = self.LSTM(embed)
        outputs = self.linear(prediction)
        return outputs

       def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_caption = []
        
        for i in range(max_len):
            
            output, (h, c) = self.LSTM(inputs, states)
            output = self.linear(output.squeeze(1))
            predicted = output.argmax(1) #in this way, we can reach index of max element
            predicted_caption.append(predicted.item())
            
            # we use predicted word as an input of lstm
            inputs = self.caption_embeddings(predicted) 
            inputs = inputs.unsqueeze(1)
        return predicted_caption