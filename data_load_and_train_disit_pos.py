import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import math
import random
import numpy as np
import csv
import copy
import pandas as pd
import re
import os

from transformer import Seq2SeqTransformer

class Config:
    DATA_PATH = 'data/addition_train_dataset_2d.csv'
    TEST_DATA_PATH = 'data/addition_test_dataset_2d.csv'
    TRAIN_RATIO = 0.8
    MAX_LEN = 30
    MAX_DIGITS = 5
    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT_P = 0.1
    LEARNING_RATE = 1.0
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    BASE_SEED = 42
    MODEL_SAVE_DIR = 'models/model_2d'
    LOG_DIR = 'log/logs_2d'
    ANALYSIS_DIR = 'analysis/analysis_2d'
    EARLY_STOPPING_PATIENCE = 10
    WARMUP_STEPS = 500
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.98
    ADAM_EPSILON = 1e-9
    LR_SCALE_FACTOR = 0.05
    LABEL_SMOOTHING = 0.1
    NUM_EXPERIMENTS = 10


SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
OPERATORS = ['+', '=']

def get_lr_scheduler(optimizer, d_model, warmup_steps, lr_scale_factor=0.05):
    d_model_scale = d_model ** -0.5

    def lr_lambda(current_step: int):
        current_step += 1
        arg1 = current_step ** -0.5
        arg2 = current_step * (warmup_steps ** -1.5)
        return lr_scale_factor * d_model_scale * min(arg1, arg2)

    return LambdaLR(optimizer, lr_lambda)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def build_vocab(data_pairs_list, max_digits):
    token_counts = {}

    base_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + OPERATORS
    for token in base_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    for digit in range(10):
        for pos_idx in range(max_digits):
            token = f"{digit}_pos{pos_idx}"
            token_counts[token] = token_counts.get(token, 0) + 1

    for data_pairs in data_pairs_list:
        if not data_pairs: continue
        for src_str, tgt_str in data_pairs:
            pass

    token_to_id = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    id_to_token = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
    next_id = 4

    for op in OPERATORS:
        if op not in token_to_id:
            token_to_id[op] = next_id
            id_to_token[next_id] = op
            next_id += 1

    digit_pos_tokens = []
    for pos_idx in range(max_digits):
        for digit in range(10):
             digit_pos_tokens.append(f"{digit}_pos{pos_idx}")

    for token in sorted(list(set(digit_pos_tokens))):
        if token not in token_to_id:
            token_to_id[token] = next_id
            id_to_token[next_id] = token
            next_id += 1

    return token_to_id, id_to_token


def tokenize_and_numericalize(text, token_to_id, max_digits):
    token_ids = [token_to_id[SOS_TOKEN]]
    parts = [p for p in re.split(r'(\d+|[^\d\s])', text) if p]

    for part in parts:
        if part.isdigit():
            num_str = part
            num_len = len(num_str)
            if num_len > max_digits:
                print(f"Warning: Number {num_str} exceeds MAX_DIGITS ({max_digits}). Treating as UNK sequence.")
                for _ in range(num_len):
                    token_ids.append(token_to_id[UNK_TOKEN])
                continue
            for i, digit_char in enumerate(num_str):
                pos_idx = num_len - 1 - i
                token = f"{digit_char}_pos{pos_idx}"
                token_ids.append(token_to_id.get(token, token_to_id[UNK_TOKEN]))
        elif part in token_to_id:
            token_ids.append(token_to_id[part])
        else:
            token_ids.append(token_to_id[UNK_TOKEN])

    token_ids.append(token_to_id[EOS_TOKEN])
    return token_ids

def ids_to_string(ids, id_to_token, sos_id, eos_id, pad_id):
    tokens = []
    current_number_parts = []

    for token_id in ids:
        if token_id == sos_id: continue
        if token_id == eos_id: break
        if token_id == pad_id: continue

        token_str = id_to_token.get(token_id, UNK_TOKEN)

        if "_pos" in token_str:
            digit = token_str.split('_')[0]
            current_number_parts.append(digit)
        else:
            if current_number_parts:
                tokens.append("".join(current_number_parts))
                current_number_parts = []
            tokens.append(token_str)

    if current_number_parts:
        tokens.append("".join(current_number_parts))

    return "".join(tokens)


class ArithmeticDataset(Dataset):
    def __init__(self, data_pairs, token_to_id, max_digits):
        self.data_pairs = data_pairs
        self.token_to_id = token_to_id
        self.max_digits = max_digits

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_str, tgt_str = self.data_pairs[idx]
        src_ids = tokenize_and_numericalize(src_str, self.token_to_id, self.max_digits)
        tgt_ids = tokenize_and_numericalize(tgt_str, self.token_to_id, self.max_digits)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def load_csv_data(file_path):
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    src_str, tgt_str = row
                    pairs.append((src_str, tgt_str))
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Returning empty list.")
    return pairs

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def collate_fn(batch, pad_id, device):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=False, padding_value=pad_id).to(device)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=False, padding_value=pad_id).to(device)
    src_padding_mask = (src_padded == pad_id).transpose(0, 1).to(device)
    tgt_padding_mask = (tgt_padded == pad_id).transpose(0, 1).to(device)
    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask

def train_epoch(model, dataloader, optimizer, criterion, device, pad_id, scheduler, current_total_steps):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_elements = 0
    current_lr = optimizer.param_groups[0]['lr']

    for src, tgt, src_padding_mask, tgt_padding_mask in dataloader:
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        tgt_input_padding_mask = tgt_padding_mask[:, :-1] if tgt_padding_mask is not None else None
        tgt_mask_causal = generate_square_subsequent_mask(tgt_input.size(0), device)

        optimizer.zero_grad()
        output_logits = model(src=src,
                              tgt=tgt_input,
                              src_mask=None,
                              tgt_mask=tgt_mask_causal,
                              src_padding_mask=src_padding_mask,
                              tgt_padding_mask=tgt_input_padding_mask,
                              memory_key_padding_mask=src_padding_mask)
        loss = criterion(output_logits.reshape(-1, output_logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        total_loss += loss.item()
        preds = output_logits.argmax(dim=-1)
        non_pad_elements = (tgt_output != pad_id)
        correct_predictions += (preds == tgt_output)[non_pad_elements].sum().item()
        total_elements += non_pad_elements.sum().item()
        current_total_steps += 1

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct_predictions / total_elements if total_elements > 0 else 0
    return avg_loss, accuracy, current_total_steps, current_lr

def evaluate_model(model, dataloader, criterion, device, id_to_token,
                   max_output_len, sos_id, eos_id, pad_id,
                   analyze_results_filename=None):
    model.eval()
    total_greedy_score = 0
    total_sequences = 0
    correct_greedy_sequences = 0
    total_val_loss = 0
    analysis_results = []

    with torch.no_grad():
        for src, tgt, src_padding_mask, tgt_padding_mask in dataloader:
            batch_size = src.size(1)
            total_sequences += batch_size

            tgt_input_val = tgt[:-1, :]
            tgt_output_val = tgt[1:, :]
            tgt_input_padding_mask_val = tgt_padding_mask[:, :-1] if tgt_padding_mask is not None else None
            tgt_mask_causal_val = generate_square_subsequent_mask(tgt_input_val.size(0), device)

            output_logits_val = model(src=src,
                                  tgt=tgt_input_val,
                                  src_mask=None,
                                  tgt_mask=tgt_mask_causal_val,
                                  src_padding_mask=src_padding_mask,
                                  tgt_padding_mask=tgt_input_padding_mask_val,
                                  memory_key_padding_mask=src_padding_mask)

            loss_val = criterion(output_logits_val.reshape(-1, output_logits_val.size(-1)), tgt_output_val.reshape(-1))
            total_val_loss += loss_val.item()

            memory = model.encode(src, src_mask=None, src_padding_mask=src_padding_mask)
            for i in range(batch_size):
                current_src_tensor = src[:, i:i+1]
                current_tgt_tensor = tgt[:, i:i+1]
                current_memory = memory[:, i:i+1, :]
                current_memory_key_padding_mask = src_padding_mask[i:i+1, :] if src_padding_mask is not None else None

                ys = torch.ones(1, 1).fill_(sos_id).type_as(src.data).to(device)
                generated_sequence_ids = [sos_id]

                for _ in range(max_output_len - 1):
                    tgt_causal_mask = generate_square_subsequent_mask(ys.size(0), device)
                    current_tgt_padding_mask = torch.zeros(ys.size(1), ys.size(0), dtype=torch.bool, device=device)

                    decoder_output = model.decode(ys, current_memory, tgt_causal_mask,
                                                  current_tgt_padding_mask, current_memory_key_padding_mask)
                    last_token_logits = model.generator(decoder_output[-1, :, :])
                    predicted_id = last_token_logits.argmax(dim=-1).item()
                    generated_sequence_ids.append(predicted_id)
                    if predicted_id == eos_id:
                        break
                    new_token_tensor = torch.ones(1, 1).type_as(src.data).fill_(predicted_id).to(device)
                    ys = torch.cat((ys, new_token_tensor), dim=0)

                input_expression = ids_to_string(current_src_tensor[:, 0].tolist(), id_to_token, sos_id, eos_id, pad_id)
                correct_answer_str = ids_to_string(current_tgt_tensor[:, 0].tolist(), id_to_token, sos_id, eos_id, pad_id)
                model_prediction_str = ids_to_string(generated_sequence_ids, id_to_token, sos_id, eos_id, pad_id)

                is_correct = (model_prediction_str == correct_answer_str)
                if is_correct:
                    total_greedy_score += 1
                    correct_greedy_sequences +=1

                analysis_results.append({
                    "Input Expression": input_expression,
                    "Correct Answer": correct_answer_str,
                    "Model Prediction": model_prediction_str,
                    "Is Correct": 1 if is_correct else 0
                })

    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy_style_2 = correct_greedy_sequences / total_sequences if total_sequences > 0 else 0

    if analyze_results_filename and analysis_results:
        try:
            df = pd.DataFrame(analysis_results)
            df.to_csv(analyze_results_filename, index=False, encoding='utf-8-sig')
            print(f"Prediction analysis saved to {analyze_results_filename}")
        except Exception as e:
            print(f"Error saving prediction analysis to {analyze_results_filename}: {e}")

    return avg_val_loss, total_greedy_score, accuracy_style_2, total_sequences


def run_experiment(experiment_id, config):
    seed_everything(config.BASE_SEED + experiment_id)

    model_save_path = os.path.join(config.MODEL_SAVE_DIR, f'model_2d_exp{experiment_id}.pth')
    log_file_path = os.path.join(config.LOG_DIR, f'training_log_2d_exp{experiment_id}.csv')
    prediction_analysis_filename = os.path.join(config.ANALYSIS_DIR, f'prediction_analysis_2d_exp{experiment_id}.csv')

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)

    print(f"\n--- Starting Experiment {experiment_id + 1}/{config.NUM_EXPERIMENTS} ---")

    train_val_data_pairs = load_csv_data(config.DATA_PATH)
    if not train_val_data_pairs:
        raise ValueError(f"No training/validation data loaded from {config.DATA_PATH}.")

    test_data_pairs = load_csv_data(config.TEST_DATA_PATH)
    if not test_data_pairs:
        print(f"Warning: No dedicated test data loaded from {config.TEST_DATA_PATH}.")

    token_to_id, id_to_token = build_vocab([train_val_data_pairs, test_data_pairs] if test_data_pairs else [train_val_data_pairs], config.MAX_DIGITS)
    vocab_size = len(token_to_id)
    PAD_ID = token_to_id[PAD_TOKEN]
    SOS_ID = token_to_id[SOS_TOKEN]
    EOS_ID = token_to_id[EOS_TOKEN]

    print(f"Vocabulary size: {vocab_size}")

    train_size = int(config.TRAIN_RATIO * len(train_val_data_pairs))
    val_size = len(train_val_data_pairs) - train_size

    train_pairs_list = list(train_val_data_pairs)
    val_pairs_list = list(train_val_data_pairs)

    if val_size > 0 and train_size > 0 :
         train_pairs_split, val_pairs_split = random_split(train_val_data_pairs, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(config.BASE_SEED + experiment_id))
         train_pairs_list = list(train_pairs_split)
         val_pairs_list = list(val_pairs_split)
    elif val_size == 0 and len(train_val_data_pairs) > 0:
        print("Warning: Not enough data to create a distinct validation set. Using all data for training and validation.")

    train_dataset = ArithmeticDataset(train_pairs_list, token_to_id, config.MAX_DIGITS)
    val_dataset = ArithmeticDataset(val_pairs_list, token_to_id, config.MAX_DIGITS)

    if test_data_pairs:
        final_test_dataset = ArithmeticDataset(test_data_pairs, token_to_id, config.MAX_DIGITS)
        final_test_dataloader = DataLoader(final_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                     collate_fn=lambda b: collate_fn(b, PAD_ID, DEVICE))
    else:
        final_test_dataloader = None

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda b: collate_fn(b, PAD_ID, DEVICE))
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                 collate_fn=lambda b: collate_fn(b, PAD_ID, DEVICE))

    model = Seq2SeqTransformer(input_vocab_size=vocab_size,
                               output_vocab_size=vocab_size,
                               d_model=config.D_MODEL,
                               nhead=config.NHEAD,
                               num_encoder_layers=config.NUM_ENCODER_LAYERS,
                               num_decoder_layers=config.NUM_DECODER_LAYERS,
                               dim_feedforward=config.DIM_FEEDFORWARD,
                               dropout_p=config.DROPOUT_P,
                               max_seq_len=config.MAX_LEN).to(DEVICE)

    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           betas=(config.ADAM_BETA1, config.ADAM_BETA2),
                           eps=config.ADAM_EPSILON)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=config.LABEL_SMOOTHING)

    lr_scheduler = get_lr_scheduler(optimizer, config.D_MODEL, config.WARMUP_STEPS,
                                    lr_scale_factor=config.LR_SCALE_FACTOR)

    experiment_train_losses = []
    experiment_val_losses = []
    experiment_train_token_accuracies = []
    experiment_val_seq_scores = []
    experiment_val_seq_accuracies = []
    experiment_learning_rates = []

    best_epoch_val_seq_accuracy = -1
    current_best_train_token_acc = -1

    epochs_no_improve = 0
    best_model_state_dict = None
    current_total_steps = 0

    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_token_acc, current_total_steps, current_lr = train_epoch(
            model, train_dataloader, optimizer, criterion, DEVICE, PAD_ID, lr_scheduler, current_total_steps
        )
        val_loss, val_score, val_seq_acc, _ = evaluate_model(
            model, val_dataloader, criterion, DEVICE, id_to_token,
            config.MAX_LEN, SOS_ID, EOS_ID, PAD_ID,
            analyze_results_filename=None
        )

        experiment_train_losses.append(train_loss)
        experiment_train_token_accuracies.append(train_token_acc)
        experiment_val_losses.append(val_loss)
        experiment_val_seq_scores.append(val_score)
        experiment_val_seq_accuracies.append(val_seq_acc)
        experiment_learning_rates.append(current_lr)

        if train_token_acc > current_best_train_token_acc:
            current_best_train_token_acc = train_token_acc

        print(f"Epoch {epoch}/{config.NUM_EPOCHS}:")
        print(f"  LR: {current_lr:.8f}")
        print(f"  Train Loss: {train_loss:.4f}, Train Token Acc: {train_token_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Seq Acc: {val_seq_acc:.4f} (Correct Sequences: {val_score})")

        if val_seq_acc > best_epoch_val_seq_accuracy:
            best_epoch_val_seq_accuracy = val_seq_acc
            epochs_no_improve = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"  Validation sequence accuracy improved to {val_seq_acc:.4f}. Saving model state for this experiment.")
        else:
            epochs_no_improve += 1
            print(f"  Validation sequence accuracy did not improve for {epochs_no_improve} epoch(s). Best for this experiment: {best_epoch_val_seq_accuracy:.4f}")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs for experiment {experiment_id + 1}.")
            break

    print(f"\nTraining finished for experiment {experiment_id + 1}.")

    if best_model_state_dict:
        print(f"Loading best model state for experiment {experiment_id + 1} for final evaluation and saving.")
        model.load_state_dict(best_model_state_dict)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model for experiment {experiment_id + 1} saved to {model_save_path}")
    else:
        print(f"No best model state was saved for experiment {experiment_id + 1}. Using last model state for saving.")
        torch.save(model.state_dict(), model_save_path)
        print(f"Last model state for experiment {experiment_id + 1} saved to {model_save_path}")


    log_header = ["epoch", "train_loss", "train_token_acc", "val_loss", "val_seq_score", "val_seq_acc", "learning_rate"]
    log_data = []
    num_logged_epochs = len(experiment_train_losses)
    for i in range(num_logged_epochs):
        log_data.append([
            i + 1,
            experiment_train_losses[i], experiment_train_token_accuracies[i],
            experiment_val_losses[i], experiment_val_seq_scores[i], experiment_val_seq_accuracies[i],
            experiment_learning_rates[i]
        ])

    try:
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
            writer.writerows(log_data)
        print(f"Training log for experiment {experiment_id + 1} saved to {log_file_path}")
    except IOError:
        print(f"Error: Could not write training log to {log_file_path}")

    test_total_correct_sequences = 0
    test_sequence_accuracy = 0.0
    if final_test_dataloader:
        print(f"\n--- Final Evaluation on Dedicated Test Set (Experiment {experiment_id + 1}) ---")
        test_loss, test_total_score, test_seq_acc, num_test_samples = evaluate_model(
            model, final_test_dataloader, criterion, DEVICE, id_to_token,
            config.MAX_LEN, SOS_ID, EOS_ID, PAD_ID,
            analyze_results_filename=prediction_analysis_filename
        )
        test_total_correct_sequences = test_total_score
        test_sequence_accuracy = test_seq_acc
        print(f"Test Set (Exp {experiment_id+1}): Total Samples: {num_test_samples}")
        print(f"Test Set (Exp {experiment_id+1}): Loss: {test_loss:.4f}")
        print(f"Test Set (Exp {experiment_id+1}): Total Correct Sequences: {test_total_correct_sequences}")
        print(f"Test Set (Exp {experiment_id+1}): Sequence Accuracy: {test_sequence_accuracy:.4f}")
    else:
        print("\nSkipping final evaluation on dedicated test set as no test data was loaded.")

    return current_best_train_token_acc, best_epoch_val_seq_accuracy, test_total_correct_sequences, test_sequence_accuracy


if __name__ == '__main__':
    all_experiments_train_token_accs = []
    all_experiments_val_seq_accs = []
    all_experiments_test_correct_sequences = []
    all_experiments_test_seq_accuracies = []

    for i in range(Config.NUM_EXPERIMENTS):
        exp_train_acc, exp_val_acc, exp_test_correct, exp_test_seq_acc = run_experiment(i, Config())

        if exp_train_acc is not None:
             all_experiments_train_token_accs.append(exp_train_acc)
        if exp_val_acc is not None:
            all_experiments_val_seq_accs.append(exp_val_acc)
        if exp_test_correct is not None:
            all_experiments_test_correct_sequences.append(exp_test_correct)
        if exp_test_seq_acc is not None:
            all_experiments_test_seq_accuracies.append(exp_test_seq_acc)

    print("\n\n--- Overall Experiment Results ---")

    if all_experiments_train_token_accs:
        best_train_token_acc_overall = max(all_experiments_train_token_accs)
        print(f"Best Train Token Acc across {Config.NUM_EXPERIMENTS} experiments: {best_train_token_acc_overall:.4f}")
    else:
        print("No Train Token Acc data to report.")

    if all_experiments_val_seq_accs:
        best_val_seq_acc_overall = max(all_experiments_val_seq_accs)
        print(f"Best Val Seq Acc across {Config.NUM_EXPERIMENTS} experiments: {best_val_seq_acc_overall:.4f}")
    else:
        print("No Val Seq Acc data to report.")

    if all_experiments_test_correct_sequences:
        avg_test_correct_sequences = np.mean(all_experiments_test_correct_sequences)
        print(f"Average Test Set Total Correct Sequences across {Config.NUM_EXPERIMENTS} experiments: {avg_test_correct_sequences:.2f}")
    else:
        print("No Test Set Total Correct Sequences data to report.")

    if all_experiments_test_seq_accuracies:
        avg_test_seq_accuracy = np.mean(all_experiments_test_seq_accuracies)
        print(f"Average Test Set Sequence Accuracy across {Config.NUM_EXPERIMENTS} experiments: {avg_test_seq_accuracy:.4f}")
    else:
        print("No Test Set Sequence Accuracy data to report.")

