from typing import Optional, Union, Tuple
import torch
from torch import nn
from transformers.utils import ModelOutput
from transformers import ElectraModel, ElectraPreTrainedModel
from dataclasses import dataclass
import numpy as np
from torch.nn import functional as F
#import wandb

device = torch.device("cuda")

@dataclass
class TokenClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
         
class extendNER(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels+2
        self.T = 0
        self.ce = 0
        self.kd = 0

        self.electra = ElectraModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        soft_labels: Optional[torch.Tensor] = None,
        hard_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
     
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)

        loss = None
        
        if hard_labels is not None and soft_labels is not None:

            output = logits.view(-1, self.num_labels)#batch*300,7
            
            batch_length = output.shape[0]
            hard_label = hard_labels.view(batch_length)#batch*300
            soft_label = soft_labels.view(batch_length, self.num_labels-2)#batch*300,5

            attention_mask = attention_mask.view(-1)#batch*300
    
            a_mask = (attention_mask>0).type(torch.BoolTensor)
            output = output[a_mask,:]
            hlabel = hard_label[a_mask]#attention_mask
            slabel = soft_label[a_mask,:]#attention_mask,5
            
            h_mask = (hlabel>0).type(torch.BoolTensor)#cls, sep 포함
            s_mask = (hlabel==0).type(torch.BoolTensor)
            #hlabel = hlabel[h_mask]-self.num_labels+2.0
            hlabel = hlabel[h_mask]
            slabel = slabel[s_mask,:]
            htoken = output[h_mask,:]
            stoken = output[s_mask,:]
            
            #import pdb;pdb.set_trace()
            
            #stoken = stoken[:,:self.num_labels-2]
            #htoken = htoken[:,self.num_labels-2:]
            
            h = htoken.tolist()
            hard_exist = True
            if not h:
                hard_exist=False
            
            ce = nn.CrossEntropyLoss()
            
            hlabel = hlabel.type(torch.LongTensor).to(device)
            
            loss_ce = 0
            loss_kd = 0
            
            if hard_exist:
                loss_ce = ce(htoken, hlabel)
            stoken = stoken / self.T
            slabel = slabel / self.T
            p_T = F.softmax(slabel, dim=-1)
            p_T = F.pad(p_T, (0,2), 'constant', 0)
            
            #loss_kd = -(p_T * F.log_softmax(stoken, dim=-1)).sum(dim=-1).mean()
            loss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(stoken, dim=-1), p_T)
            
            loss = self.ce*loss_ce + self.kd*loss_kd
            #wandb.log({"ce_loss": loss_ce, "kd_loss": loss_kd})
            #print("loss_ce : {}, loss_kd : {}".format(loss_ce, loss_kd))
            

        else:
            output = logits.view(-1, self.num_labels)
            batch_length = output.shape[0]
            labels = labels.view(batch_length)
            ce = nn.CrossEntropyLoss()
            loss = ce(output, labels)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output
        
 
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions
        )

