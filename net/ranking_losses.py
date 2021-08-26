import torch


def rank_sort_loss(ctx, mask_logits, gt_labels, delta_RS=0.50, eps=1e-10): 

    classification_grads=torch.zeros(logits.shape).cuda()
        
    #Filter fg logits
    fg_labels = (targets > 0.)
    fg_logits = logits[fg_labels]
    fg_targets = targets[fg_labels]
    fg_num = len(fg_logits)

    #Do not use bg with scores less than minimum fg logit
    #since changing its score does not have an effect on precision
    threshold_logit = torch.min(fg_logits)-delta_RS
    relevant_bg_labels=((targets==0) & (logits>=threshold_logit))
        
    relevant_bg_logits = logits[relevant_bg_labels] 
    relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
    sorting_error=torch.zeros(fg_num).cuda()
    ranking_error=torch.zeros(fg_num).cuda()
    fg_grad=torch.zeros(fg_num).cuda()
        
    #sort the fg logits
    order=torch.argsort(fg_logits)
    #Loops over each positive following the order
    for ii in order:
        # Difference Transforms (x_ij)
        fg_relations=fg_logits-fg_logits[ii] 
        bg_relations=relevant_bg_logits-fg_logits[ii]

        if delta_RS > 0:
            fg_relations=torch.clamp(fg_relations/(2*delta_RS)+0.5,min=0,max=1)
            bg_relations=torch.clamp(bg_relations/(2*delta_RS)+0.5,min=0,max=1)
        else:
            fg_relations = (fg_relations >= 0).float()
            bg_relations = (bg_relations >= 0).float()

        # Rank of ii among pos and false positive number (bg with larger scores)
        rank_pos=torch.sum(fg_relations)
        FP_num=torch.sum(bg_relations)

        # Rank of ii among all examples
        rank=rank_pos+FP_num
                            
        # Ranking error of example ii. target_ranking_error is always 0. (Eq. 7)
        ranking_error[ii]=FP_num/rank      

        # Current sorting error of example ii. (Eq. 7)
        current_sorting_error = torch.sum(fg_relations*(1-fg_targets))/rank_pos

        #Find examples in the target sorted order for example ii         
        iou_relations = (fg_targets >= fg_targets[ii])
        target_sorted_order = iou_relations * fg_relations

        #The rank of ii among positives in sorted order
        rank_pos_target = torch.sum(target_sorted_order)

        #Compute target sorting error. (Eq. 8)
        #Since target ranking error is 0, this is also total target error 
        target_sorting_error= torch.sum(target_sorted_order*(1-fg_targets))/rank_pos_target

        #Compute sorting error on example ii
        sorting_error[ii] = current_sorting_error - target_sorting_error
  
        #Identity Update for Ranking Error 
        if FP_num > eps:
            #For ii the update is the ranking error
            fg_grad[ii] -= ranking_error[ii]
            #For negatives, distribute error via ranking pmf (i.e. bg_relations/FP_num)
            relevant_bg_grad += (bg_relations*(ranking_error[ii]/FP_num))

        #Find the positives that are misranked (the cause of the error)
        #These are the ones with smaller IoU but larger logits
        missorted_examples = (~ iou_relations) * fg_relations

        #Denominotor of sorting pmf 
        sorting_pmf_denom = torch.sum(missorted_examples)

        #Identity Update for Sorting Error 
        if sorting_pmf_denom > eps:
            #For ii the update is the sorting error
            fg_grad[ii] -= sorting_error[ii]
            #For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
            fg_grad += (missorted_examples*(sorting_error[ii]/sorting_pmf_denom))

    #Normalize gradients by number of positives 
    classification_grads[fg_labels]= (fg_grad/fg_num)
    classification_grads[relevant_bg_labels]= (relevant_bg_grad/fg_num)

    ctx.save_for_backward(classification_grads)

    return ranking_error.mean(), sorting_error.mean()

@staticmethod
def backward(ctx, out_grad1, out_grad2):
    g1, =ctx.saved_tensors
    return g1*out_grad1, None, None, None

