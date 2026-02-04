import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoTokenizer
import re



def preprocess_distillation_batch(batch):
    batch_dict = {"student_" + key: value for key, value in batch[0].items()}
    batch_dict.update({"teacher_" + key: value for key,
                      value in batch[1].items()})
    return batch_dict

def improved_sort(value):
    sums = value.sum(dim=(0, 1))
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_values = value[:, :, sorted_indices]
    return sorted_values

def normalize(value):
    means = value.mean(dim=-1, keepdim=True)
    stds = value.std(dim=-1, keepdim=True)
    z_score_normalized_student = (value)/ (stds+0.0001)
    return z_score_normalized_student

def KL_wo(y_s, y_t,T=1):
    p_s = F.log_softmax(y_s/T, dim=-1)
    p_t = F.softmax(y_t/T, dim=-1)
    loss = -torch.sum(p_t * p_s, dim=-1).mean()
    return loss

class Sinkhorn_seq(nn.Module):
    def __init__(self, T=2):
        super(Sinkhorn_seq, self).__init__()
        self.T = 2   
    def sinkhorn_normalized(self,x, n_iters=20):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self,x, y, epsilon=0.1, n_iters=10):
        Wxy = torch.cdist(x, y, p=1)  
        K = torch.exp(-Wxy / epsilon)  
        P = self.sinkhorn_normalized(K, n_iters)  
        return torch.sum(P * Wxy)  
    def forward(self, y_s, y_t):
        softmax = nn.Softmax(dim=-1)
        p_s = softmax(y_s/self.T)
        p_t = softmax(y_t/self.T)
        emd_loss = 0
        for i in range(p_s.shape[0]):
            emd_loss += 0.001*self.sinkhorn_loss(x=p_s[i],y=p_t[i])
        return emd_loss

def greedy_algorithm_adjust_s(t, s):
    batch_size, T, k = t.shape
    _, n, _ = s.shape
    
    # Initialize the adjusted source tensor
    s_adjusted = torch.zeros_like(t)
    
    for b in range(batch_size):
        # Initialize set of available source indices for each batch
        available_indices = list(range(n))
        
        for i in range(T):
            C_min = float('inf')
            j_star = -1
            
            for j in available_indices:
                # Compute cost as the sum of absolute differences for each batch
                C = torch.sum(torch.abs(t[b,:,i] - s[b,:,j]))
                
                if C < C_min:
                    C_min = C
                    j_star = j
            
            # Assign the best matching source vector to the adjusted tensor
            s_adjusted[b,:,i] = s[b,:,j_star]
            
            # Remove the selected index from available indices
            available_indices.remove(j_star)

    return s_adjusted

class DistillationModel(nn.Module):
    def __init__(self, student, teacher, teacher_tokenizer, student_tokenizer):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()

    def forward(self, student_input_ids, student_attention_mask, student_labels, teacher_input_ids, teacher_attention_mask, teacher_labels):
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                labels=teacher_labels
            )

        student_output = self.student(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=student_labels
        )
        return student_output, teacher_output


# change stu label to teacher generation

class DistillationModel2(nn.Module):
    def __init__(self, student, teacher, teacher_tokenizer, student_tokenizer, ignore_index=-100):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.teacher.eval()
        self.ignore_index = ignore_index

    def forward(self, student_input_ids, student_attention_mask, student_labels, teacher_input_ids, teacher_attention_mask, teacher_labels):
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                labels=teacher_labels
            )
            teacher_logits = teacher_output.logits
            teacher_token_ids = torch.argmax(teacher_logits, dim=-1)

        student_answer_index, student_answer_size = self.__get_start_and_size_answers(student_labels)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(teacher_labels)


        teacher_answers_text = []
        for i in range(len(teacher_answer_index)):
            start_idx = teacher_answer_index[i]
            end_idx = start_idx + teacher_answer_size[i]
            answer_ids = teacher_token_ids[i, start_idx:end_idx] 
            answer_text = self.teacher_tokenizer.decode(answer_ids)  
            teacher_answers_text.append(answer_text)

        new_student_labels = torch.full_like(student_labels, fill_value=-100) 
        for i, answer_text in enumerate(teacher_answers_text):
            student_start_idx = student_answer_index[i]
            encoded_answer = self.student_tokenizer.encode(answer_text, add_special_tokens=True)  
            end_idx = min(student_start_idx + len(encoded_answer), student_labels.size(1))  
            new_student_labels[i, student_start_idx:end_idx] = torch.tensor(encoded_answer[:end_idx-student_start_idx], dtype=torch.long)

        #print(teacher_answers_text)
        #print(student_labels)
        #print(new_student_labels)


        

        student_output = self.student(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=new_student_labels
        )
        return student_output, teacher_output
    
    def __get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            is_value = answer.eq(self.ignore_index)
            answers_size.append(len(answer) - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
        return answers_index, answers_size


class DistillationLoss(nn.Module):
    def __init__(self, batch_limit=100, store_path='teacher_logits_partial.npy', crossentropy_weight=1, distillation_weight=1, student_temperature=1, teacher_temperature=1, skip_student_eos=False, skip_teacher_eos=False, ignore_index=-100, debug=False, debug_rank=0, tokenizer_student=None, tokenizer_teacher=None, f=1):
        super().__init__()
        self.crossentropy_weight = crossentropy_weight
        self.distillation_weight = distillation_weight
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.ignore_index = ignore_index
        self.debug_rank = debug_rank
        self.debug = debug
        self.f = f
        
        self.store_teacher_logits = True
        self.batch_limit = batch_limit  # 设定每100个样本保存一次
        self.store_path = store_path
        self.teacher_logits_temp_storage = []

        if self.debug:
            print("Distillation loss parameters:")
            print(f"Crossentropy weight: {crossentropy_weight}")
            print(f"Distillation weight: {distillation_weight}")
            print(f"Student temperature: {student_temperature}")
            print(f"Teacher temperature: {teacher_temperature}")
            print(f"Skip student eos: {skip_student_eos}")
            print(f"Skip teacher eos: {skip_teacher_eos}")
            print(f"Ignore index: {ignore_index}")
            print(f"Debug: {debug}")
            print(f"Debug rank: {debug_rank}")

            self.student_tokenizer = AutoTokenizer.from_pretrained(tokenizer_student,trust_remote_code=True)
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(tokenizer_teacher,trust_remote_code=True)

    def forward(self, epoch, student_predictions, teacher_predictions, student_targets, teacher_targets, rank=0):
        student = student_predictions.logits
        teacher = teacher_predictions.logits
        if self.store_teacher_logits:
            processed_teacher_logits = []

        # Get answer first token and answer size
        student_answer_index, student_answer_size = self.__get_start_and_size_answers(
            student_targets)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(
            teacher_targets)

        # Avoid eos token, if needed
        if self.skip_student_eos: student_answer_size = [size-1 for size in student_answer_size]
        if self.skip_teacher_eos: teacher_answer_size = [size-1 for size in teacher_answer_size]
        
        student = normalize(student)      
        teacher = normalize(teacher)

        # Align answer first token, pad to right and compute softmax
        for i in range(student.size(0)):
            shift = student_answer_index[i]
            size = student_answer_size[i]
            end_shift = shift+size
            student[i] = torch.cat((
                torch.nn.functional.softmax(student[i, shift:end_shift, :]/self.student_temperature, dim=-1),
                torch.zeros_like(student[i, :(student.size(1)-size), :])), dim=0
            )
        for i in range(teacher.size(0)):
            shift = teacher_answer_index[i]
            size = teacher_answer_size[i]
            end_shift = shift+size
            teacher[i] = torch.cat((
               torch.nn.functional.softmax(teacher[i, shift:end_shift, :]/self.teacher_temperature, dim=-1),
               torch.zeros_like(teacher[i, :(teacher.size(1)-size), :])), dim=0
            )
        
        # Cut to max answer length
        mex_length = max(max(student_answer_size), max(teacher_answer_size))

        student = student[:, :mex_length, :]
        teacher = teacher[:, :mex_length, :]
        
        sinkorn_loss = Sinkhorn_seq()

        if self.debug and rank == self.debug_rank:
            print("\n\n----------------------------------")
            print("------- Label / Prediction -------")
            print("----------------------------------")
            student_labels = [row[row != -100] for row in student_targets]
            teacher_labels = [row[row != -100] for row in teacher_targets]
            print("------- Label shape -------")
            print(f"Student label shape: {student_answer_size[0]}")
            print(f"Teacher label shape: {teacher_answer_size[0]}")
            print("------- Student Label -> Prediction -------")
            print(self.student_tokenizer.batch_decode(student_labels[0]))
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Teacher Label -> Prediction -------")
            print(self.teacher_tokenizer.batch_decode(teacher_labels[0]))
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            print("------- Prediction Teacher -> Student  -------")
            
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Shape -------")
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}\n")

        # # Sort in descending order to align probabilities
        # student = student.sort(dim=-1, descending=True).values
        # teacher = teacher.sort(dim=-1, descending=True).values
        teacher = improved_sort(teacher)
        teacher = teacher[:,:,:50]
        if self.f == 1:
            student = improved_sort(student)
            student = student[:,:,:50]
        elif self.f == 2:
            student = greedy_algorithm_adjust_s(teacher,student)

        # Pad to get same vocabulary size
        diff_size = student.size(2) - teacher.size(2)
        if diff_size > 0:
            teacher = F.pad(teacher, (0, diff_size), value=0)
        elif diff_size < 0:
            student = F.pad(student, (0, abs(diff_size)), value=0)
            
        if self.debug and rank == self.debug_rank:
            print("--------------------------------------------")
            print("---- Post-treatment tensor architecture ----")
            print("--------------------------------------------")
            print("------- Shape -------")
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}")
            print(" ------- First token -------")
            print(f"Student first logits: {student[0][0][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][0][:5].tolist()}")
            print(f"Student last logits: {student[0][0][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][0][-5:].tolist()}")
            print(" ------- Last token -------")
            print(f"Student first logits: {student[0][-1][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][-1][:5].tolist()}")
            print(f"Student last logits: {student[0][-1][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][-1][-5:].tolist()}\n")

        # Cross entropy loss
        crossentropy_loss = self.crossentropy_weight * student_predictions.loss

        distillation_loss = torch.zeros(student.size(0), device=student.device) 
        for i in range(student.size(0)):
            size = min(student_answer_size[i], teacher_answer_size[i])
            distillation_loss[i] = abs(student[i][:size] - teacher[i][:size]).sum(-1).mean(-1) 

        distillation_loss = distillation_loss + KL_wo(teacher,student)*0.1
        distillation_loss = distillation_loss.mean() + sinkorn_loss(teacher,student)*0.1
        distillation_loss = self.distillation_weight * (distillation_loss) * 1

        if self.debug and rank == self.debug_rank:
            print("--------------------------------------")
            print("---------------- Loss ----------------")
            print("--------------------------------------")
            print(f"Crossentropy loss: {crossentropy_loss}")
            print(f"Distillation loss: {distillation_loss}")
            print(f"Total loss: {crossentropy_loss + distillation_loss}")

        return crossentropy_loss + distillation_loss, crossentropy_loss, distillation_loss

    def __get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            is_value = answer.eq(self.ignore_index)
            answers_size.append(len(answer) - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
        return answers_index, answers_size
    
    def save_teacher_logits_partial(self):

        with open(self.store_path, 'ab') as f:  
            for logits in self.teacher_logits_temp_storage:
                np.save(f, logits)

    def on_epoch_end(self):

        if self.teacher_logits_temp_storage:
            self.save_teacher_logits_partial()
            self.teacher_logits_temp_storage = []  
