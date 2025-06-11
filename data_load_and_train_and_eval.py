import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import math
import random
import numpy as np
import csv
import copy

from transformer import Seq2SeqTransformer

class Config:
    DATA_PATH = 'data/addition_train_dataset.csv' # 修正: generate_data.py に合わせる
    TEST_DATA_PATH = 'data/addition_test_dataset.csv' # 修正: generate_data.py に合わせる
    TRAIN_RATIO = 0.8
    MAX_LEN = 20
    D_MODEL = 1024       # ユーザーがアップロードしたファイルの値に合わせる
    NHEAD = 16           # ユーザーがアップロードしたファイルの値に合わせる
    NUM_ENCODER_LAYERS = 6 # ユーザーがアップロードしたファイルの値に合わせる
    NUM_DECODER_LAYERS = 6 # ユーザーがアップロードしたファイルの値に合わせる
    DIM_FEEDFORWARD = 4096 # ユーザーがアップロードしたファイルの値に合わせる
    DROPOUT_P = 0.3     # ユーザーがアップロードしたファイルの値に合わせる
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64     # ユーザーがアップロードしたファイルの値に合わせる
    NUM_EPOCHS = 200    # ユーザーがアップロードしたファイルの値に合わせる
    SEED = 42
    MODEL_SAVE_PATH = 'seq2seq_transformer_addition_best.pth'
    EARLY_STOPPING_PATIENCE = 200 # Patienceの値を調整しやすくする (例: 10)
    LOG_FILE_PATH = 'log/training_log_model_large.csv' # 学習ログのファイルパスを追加

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(Config.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

def build_vocab(data_pairs_list):
    char_counts = {}
    for data_pairs in data_pairs_list:
        for src_str, tgt_str in data_pairs:
            for char in src_str:
                char_counts[char] = char_counts.get(char, 0) + 1
            for char in tgt_str:
                char_counts[char] = char_counts.get(char, 0) + 1
    sorted_chars = sorted(char_counts.keys())
    char_to_id = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    id_to_char = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
    for i, char in enumerate(sorted_chars):
        char_to_id[char] = i + 4
        id_to_char[i + 4] = char
    return char_to_id, id_to_char

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

def tokenize_and_numericalize(text, char_to_id, max_len):
    tokens = [char_to_id[SOS_TOKEN]]
    for char in text:
        tokens.append(char_to_id.get(char, char_to_id[UNK_TOKEN]))
    tokens.append(char_to_id[EOS_TOKEN])
    return tokens

class ArithmeticDataset(Dataset):
    def __init__(self, data_pairs, char_to_id, max_len):
        self.data_pairs = data_pairs
        self.char_to_id = char_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_str, tgt_str = self.data_pairs[idx]
        src_ids = tokenize_and_numericalize(src_str, self.char_to_id, self.max_len)
        tgt_ids = tokenize_and_numericalize(tgt_str, self.char_to_id, self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

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

def train_epoch(model, dataloader, optimizer, criterion, device, pad_id): # char_to_id は不要なので削除
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_elements = 0
    for src, tgt, src_padding_mask, tgt_padding_mask in dataloader:
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        tgt_input_padding_mask = tgt_padding_mask[:, :-1]
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
        total_loss += loss.item()
        preds = output_logits.argmax(dim=-1)
        non_pad_elements = (tgt_output != pad_id)
        correct_predictions += (preds == tgt_output)[non_pad_elements].sum().item()
        total_elements += non_pad_elements.sum().item()
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct_predictions / total_elements if total_elements > 0 else 0
    return avg_loss, accuracy

def evaluate_model(model, dataloader, criterion, device, id_to_char, max_output_len, sos_id, eos_id, pad_id):
    model.eval()
    total_greedy_score = 0
    total_sequences = 0
    correct_greedy_sequences = 0
    total_val_loss = 0

    with torch.no_grad():
        for src, tgt, src_padding_mask, tgt_padding_mask in dataloader: # tgt_padding_mask を使用
            batch_size = src.size(1)
            total_sequences += batch_size

            # --- Validation Loss Calculation (Teacher Forcing) ---
            tgt_input_val = tgt[:-1, :]
            tgt_output_val = tgt[1:, :]
            tgt_input_padding_mask_val = tgt_padding_mask[:, :-1]
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

            # --- Greedy Search for Sequence Accuracy ---
            memory = model.encode(src, src_mask=None, src_padding_mask=src_padding_mask)
            for i in range(batch_size):
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

                target_str_ids_for_comparison = [val.item() for val in tgt[1:-1, i]]
                if generated_sequence_ids and generated_sequence_ids[0] == sos_id:
                    cleaned_generated_ids = generated_sequence_ids[1:]
                else:
                    cleaned_generated_ids = generated_sequence_ids
                if cleaned_generated_ids and cleaned_generated_ids[-1] == eos_id:
                    cleaned_generated_ids = cleaned_generated_ids[:-1]
                if cleaned_generated_ids == target_str_ids_for_comparison:
                    total_greedy_score += 1
                    correct_greedy_sequences +=1
    
    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy_style_1 = total_greedy_score / 1000.0 # As per original request, though less standard
    accuracy_style_2 = correct_greedy_sequences / total_sequences if total_sequences > 0 else 0
    
    return avg_val_loss, total_greedy_score, accuracy_style_1, accuracy_style_2, total_sequences


if __name__ == '__main__':
    train_val_data_pairs = load_csv_data(Config.DATA_PATH)
    if not train_val_data_pairs:
        raise ValueError(f"No training/validation data loaded from {Config.DATA_PATH}. Please generate it first (e.g., using generate_data.py).")

    test_data_pairs = load_csv_data(Config.TEST_DATA_PATH)
    if not test_data_pairs:
        print(f"Warning: No dedicated test data loaded from {Config.TEST_DATA_PATH}. Final test evaluation will be skipped.")

    char_to_id, id_to_char = build_vocab([train_val_data_pairs, test_data_pairs] if test_data_pairs else [train_val_data_pairs])
    vocab_size = len(char_to_id)
    PAD_ID = char_to_id[PAD_TOKEN]
    SOS_ID = char_to_id[SOS_TOKEN]
    EOS_ID = char_to_id[EOS_TOKEN]

    print(f"Vocabulary size: {vocab_size}")

    train_size = int(Config.TRAIN_RATIO * len(train_val_data_pairs))
    val_size = len(train_val_data_pairs) - train_size
    if val_size == 0 and len(train_val_data_pairs) > 0 : # Ensure val_size is not zero if train_val_data_pairs is not empty
        if train_size > 1: # Cannot split if only 1 sample
             train_size -=1
             val_size +=1
        else: # Not enough data to split for validation
            print("Warning: Not enough data to create a validation set. Using training data for validation (not recommended).")
            val_pairs = list(train_val_data_pairs) # Use all as validation as well if no split possible
            train_pairs = list(train_val_data_pairs)

    if val_size > 0 :
        train_pairs, val_pairs = random_split(train_val_data_pairs, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(Config.SEED))
        train_dataset = ArithmeticDataset(list(train_pairs), char_to_id, Config.MAX_LEN)
        val_dataset = ArithmeticDataset(list(val_pairs), char_to_id, Config.MAX_LEN)
    else: # Handle case where no validation split occurs
        train_dataset = ArithmeticDataset(list(train_val_data_pairs), char_to_id, Config.MAX_LEN)
        val_dataset = train_dataset # Fallback: use train set as val set (not ideal)


    if test_data_pairs:
        final_test_dataset = ArithmeticDataset(test_data_pairs, char_to_id, Config.MAX_LEN)
        final_test_dataloader = DataLoader(final_test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                                     collate_fn=lambda b: collate_fn(b, PAD_ID, DEVICE))
    else:
        final_test_dataloader = None

    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda b: collate_fn(b, PAD_ID, DEVICE))
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                                 collate_fn=lambda b: collate_fn(b, PAD_ID, DEVICE))

    model = Seq2SeqTransformer(input_vocab_size=vocab_size,
                               output_vocab_size=vocab_size,
                               d_model=Config.D_MODEL,
                               nhead=Config.NHEAD,
                               num_encoder_layers=Config.NUM_ENCODER_LAYERS,
                               num_decoder_layers=Config.NUM_DECODER_LAYERS,
                               dim_feedforward=Config.DIM_FEEDFORWARD,
                               dropout_p=Config.DROPOUT_P,
                               max_seq_len=Config.MAX_LEN).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # --- 学習曲線記録用リスト ---
    train_losses = []
    val_losses = []
    train_token_accuracies = []
    val_seq_scores = []
    val_seq_accuracies = [] # 通常の検証シーケンス正解率

    best_val_score = -1 # Early stopping はスコア(正解数)を監視
    epochs_no_improve = 0
    best_model_state_dict = None

    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs with Early Stopping (patience={Config.EARLY_STOPPING_PATIENCE})...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss, train_token_acc = train_epoch(model, train_dataloader, optimizer, criterion, DEVICE, PAD_ID)
        val_loss, val_score, val_acc_style1, val_acc_style2, _ = evaluate_model(model, val_dataloader, criterion, DEVICE, id_to_char, Config.MAX_LEN, SOS_ID, EOS_ID, PAD_ID)
        
        train_losses.append(train_loss)
        train_token_accuracies.append(train_token_acc)
        val_losses.append(val_loss)
        val_seq_scores.append(val_score)
        val_seq_accuracies.append(val_acc_style2) # 通常の検証シーケンス正解率を記録

        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Token Acc: {train_token_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Seq Acc: {val_acc_style2:.4f} (Score: {val_score})")

        if val_score > best_val_score: # スコア(正解数)で改善を判断
            best_val_score = val_score
            epochs_no_improve = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"  Validation score improved to {val_score}. Saving model state.")
        else:
            epochs_no_improve += 1
            print(f"  Validation score did not improve for {epochs_no_improve} epoch(s). Best score: {best_val_score}")

        if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs due to no improvement for {Config.EARLY_STOPPING_PATIENCE} consecutive epochs.")
            break
            
    print("\nTraining finished.")

    if best_model_state_dict:
        print("Loading best model state for final evaluation and saving.")
        model.load_state_dict(best_model_state_dict)
    else:
        print("No best model state was saved. Using last model state.")

    print(f"\nSaving model to {Config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("Model saved.")

    # --- 学習ログをCSVに保存 ---
    log_header = ["epoch", "train_loss", "train_token_acc", "val_loss", "val_seq_score", "val_seq_acc"]
    log_data = []
    for i in range(len(train_losses)): # epoch は1から始まるので i+1
        log_data.append([i + 1, train_losses[i], train_token_accuracies[i], val_losses[i], val_seq_scores[i], val_seq_accuracies[i]])
    
    try:
        with open(Config.LOG_FILE_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
            writer.writerows(log_data)
        print(f"Training log saved to {Config.LOG_FILE_PATH}")
    except IOError:
        print(f"Error: Could not write training log to {Config.LOG_FILE_PATH}")


    if final_test_dataloader:
        print("\n--- Final Evaluation on Dedicated Test Set (with best model) ---")
        # evaluate_model を使用
        test_loss, test_total_score, test_accuracy_style_1, test_accuracy_style_2, num_test_samples = evaluate_model(
            model, final_test_dataloader, criterion, DEVICE, id_to_char,
            Config.MAX_LEN, SOS_ID, EOS_ID, PAD_ID
        )
        print(f"Test Set: Total Samples: {num_test_samples}")
        print(f"Test Set: Loss: {test_loss:.4f}") # テスト損失も表示
        print(f"Test Set: Total Score (Correct Sequences): {test_total_score}")
        print(f"Test Set: Accuracy (Total Score / 1000): {test_accuracy_style_1:.4f}")
        print(f"Test Set: Accuracy (Correct Sequences / Total Samples): {test_accuracy_style_2:.4f}")
    else:
        print("\nSkipping final evaluation on dedicated test set as no test data was loaded.")
    
    print("\n--- Inference Example (with best model) ---")
    model.eval()
    with torch.no_grad():
        test_eq = "05+03="
        src_tensor = torch.tensor(tokenize_and_numericalize(test_eq, char_to_id, Config.MAX_LEN), dtype=torch.long).unsqueeze(1).to(DEVICE)
        src_padding_mask_infer = (src_tensor == PAD_ID).transpose(0,1).to(DEVICE)

        memory_infer = model.encode(src_tensor, src_mask=None, src_padding_mask=src_padding_mask_infer)
        current_memory_key_padding_mask_infer = src_padding_mask_infer

        generated_ids_infer = [SOS_ID]
        ys_infer = torch.ones(1, 1).fill_(SOS_ID).type_as(src_tensor.data).to(DEVICE)

        for _ in range(Config.MAX_LEN -1):
            tgt_causal_mask_infer = generate_square_subsequent_mask(ys_infer.size(0), DEVICE)
            current_tgt_padding_mask_infer = torch.zeros(ys_infer.size(1), ys_infer.size(0), dtype=torch.bool, device=DEVICE)

            decoder_output_infer = model.decode(ys_infer, memory_infer[:,0:1,:], tgt_causal_mask_infer,
                                                current_tgt_padding_mask_infer, current_memory_key_padding_mask_infer)
            
            last_token_logits_infer = model.generator(decoder_output_infer[-1, :, :])
            next_token_id_infer = last_token_logits_infer.argmax(dim=-1).item()
            generated_ids_infer.append(next_token_id_infer)

            if next_token_id_infer == EOS_ID:
                break
            
            new_token_tensor_infer = torch.ones(1, 1).type_as(src_tensor.data).fill_(next_token_id_infer).to(DEVICE)
            ys_infer = torch.cat((ys_infer, new_token_tensor_infer), dim=0)
        
        result_chars = [id_to_char.get(gid, UNK_TOKEN) for gid in generated_ids_infer[1:-1]]
        print(f"Input: {test_eq}")
        print(f"Predicted: {''.join(result_chars)}")