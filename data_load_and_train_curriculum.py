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
import os # ファイルパス操作のため
from typing import List, Tuple, Set, Optional

from transformer import Seq2SeqTransformer # transformer.py が同じディレクトリにある想定

class Config:
    # --- カリキュラムラーニング用設定 ---
    # 各ステージの訓練データファイルパスのリスト
    CURRICULUM_DATA_PATHS = [
        'data/curriculum_stage1_1d_1d_nocarry.csv',
        'data/curriculum_stage2_1d_1d_carry.csv',
        'data/curriculum_stage3_2d_1d_nocarry.csv',
        'data/curriculum_stage4_2d_1d_carry.csv',
        'data/curriculum_stage5a_2d_2d_res2digit.csv', # または stage5b
        'data/curriculum_stage6_2d_2d_res3digit_carry.csv',
        # 必要に応じて 'data/addition_train_decomposed_commutative_dataset.csv' のような
        # 最終的な目標タスクのデータもここに追加できます。
    ]
    # 各ステージで学習するエポック数のリスト (CURRICULUM_DATA_PATHS と同じ長さであること)
    CURRICULUM_EPOCHS_PER_STAGE = [
        200, # ステージ1のエポック数
        200, # ステージ2のエポック数
        200, # ステージ3のエポック数
        200, # ステージ4のエポック数
        200, # ステージ5のエポック数
        200  # ステージ6のエポック数
        # 合計 NUM_EPOCHS はこのリストの合計になるか、別途最大エポック数を設ける
    ]
    # --- ここまでカリキュラムラーニング用設定 ---

    VALIDATION_DATA_PATH = 'data/addition_test_decomposed_commutative_dataset.csv' # 検証用データパス (全ステージ共通またはステージごと)
                                                                            # 今回は全ステージ共通の検証セットを使用する想定
    FINAL_TEST_DATA_PATH = 'data/addition_test_decomposed_commutative_dataset.csv' # 最終テスト用データパス

    # MAX_LEN やモデルパラメータは、最も複雑なステージのデータも扱えるように設定
    MAX_LEN = 30 # 分解形式や長い式も考慮して少し増やす (例)
    D_MODEL = 256       # 例: 縮小モデル
    NHEAD = 4
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 1024
    DROPOUT_P = 0.1
    
    LEARNING_RATE = 1.0 # スケジューラで制御
    BATCH_SIZE = 64
    # NUM_EPOCHS は CURRICULUM_EPOCHS_PER_STAGE の合計になるので、ここではコメントアウトするか、
    # 全体の最大エポック数として別途定義してもよい。
    # NUM_EPOCHS = sum(CURRICULUM_EPOCHS_PER_STAGE) # または固定の最大値

    SEED = 42
    MODEL_SAVE_PATH = 'seq2seq_transformer_curriculum_best.pth'
    EARLY_STOPPING_PATIENCE = 1200 # ステージごと、または全体で適用するか検討。今回は全体で適用。
    LOG_FILE_PATH = 'log/training_log_curriculum.csv'
    PREDICTION_ANALYSIS_FILENAME = "analysis/prediction_analysis_curriculum.csv"

    WARMUP_STEPS = 500 # 全体のステップ数に対するウォームアップ
    LR_SCALE_FACTOR = 0.035 # D_MODEL=256 の場合の調整済みスケールファクターの例
    
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.98
    ADAM_EPSILON = 1e-9
    LABEL_SMOOTHING = 0.1 # ラベルスムージングを導入


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

seed_everything(Config.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

def build_vocab(data_paths_list: List[List[Tuple[str, str]]]): # 複数のデータペアリストのリストを受け取る
    char_counts = {}
    for data_pairs_for_stage in data_paths_list:
        if not data_pairs_for_stage: continue
        for src_str, tgt_str in data_pairs_for_stage:
            for char in src_str:
                char_counts[char] = char_counts.get(char, 0) + 1
            for char in tgt_str:
                char_counts[char] = char_counts.get(char, 0) + 1
    sorted_chars = sorted(char_counts.keys())
    char_to_id = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    id_to_char = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
    next_id = 4
    for char in sorted_chars:
        if char not in char_to_id:
            char_to_id[char] = next_id
            id_to_char[next_id] = char
            next_id +=1
    return char_to_id, id_to_char

def load_csv_data(file_path):
    pairs = []
    if not os.path.exists(file_path):
        print(f"Warning: Data file not found at {file_path}. Returning empty list.")
        return pairs
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    src_str, tgt_str = row
                    pairs.append((src_str, tgt_str))
    except Exception as e:
        print(f"Error reading {file_path}: {e}. Returning empty list.")
    return pairs

def tokenize_and_numericalize(text, char_to_id):
    tokens = [char_to_id[SOS_TOKEN]]
    for char in text:
        tokens.append(char_to_id.get(char, char_to_id[UNK_TOKEN]))
    tokens.append(char_to_id[EOS_TOKEN])
    return tokens

class ArithmeticDataset(Dataset):
    def __init__(self, data_pairs, char_to_id):
        self.data_pairs = data_pairs
        self.char_to_id = char_to_id

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_str, tgt_str = self.data_pairs[idx]
        src_ids = tokenize_and_numericalize(src_str, self.char_to_id)
        tgt_ids = tokenize_and_numericalize(tgt_str, self.char_to_id)
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
    
    if not src_batch or not tgt_batch: # バッチが空の場合の対処
        return None, None, None, None

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

    if len(dataloader) == 0: # データローダーが空の場合
        print("Warning: Training dataloader is empty for this epoch.")
        return 0, 0, current_total_steps, current_lr


    for batch_idx, (src, tgt, src_padding_mask, tgt_padding_mask) in enumerate(dataloader):
        if src is None: # collate_fnがNoneを返した場合のスキップ
            print(f"Warning: Skipping empty batch at index {batch_idx}.")
            continue

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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
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

def evaluate_model(model, dataloader, criterion, device, id_to_char, char_to_id,
                   max_output_len, sos_id, eos_id, pad_id, 
                   analyze_results_filename=None):
    model.eval()
    total_greedy_score = 0
    total_sequences = 0
    correct_greedy_sequences = 0
    total_val_loss = 0
    analysis_results = []

    if len(dataloader) == 0: # データローダーが空の場合
        print("Warning: Evaluation dataloader is empty.")
        if analyze_results_filename: # 空の分析ファイルを作成または何もしない
            try:
                df = pd.DataFrame(columns=["Input Expression", "Correct Answer", "Model Prediction", "Is Correct"])
                df.to_csv(analyze_results_filename, index=False, encoding='utf-8-sig')
                print(f"Empty prediction analysis saved to {analyze_results_filename}")
            except Exception as e:
                print(f"Error saving empty prediction analysis: {e}")
        return 0, 0, 0, 0, 0


    with torch.no_grad():
        for src, tgt, src_padding_mask, tgt_padding_mask in dataloader:
            if src is None: continue # collate_fnがNoneを返した場合

            batch_size = src.size(1)
            total_sequences += batch_size

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

                src_str_list = []
                raw_src_ids = current_src_tensor[:, 0].tolist()
                start_index = 1 if raw_src_ids and raw_src_ids[0] == sos_id else 0
                end_index = len(raw_src_ids)
                if eos_id in raw_src_ids:
                    try:
                        # SOSの後の最初のEOSを探す
                        temp_end_index = raw_src_ids.index(eos_id, start_index)
                        # もしそのEOSが式の一部であるなら（例: A+B=EOS のような形式）、保持する。
                        # 今回のケースでは、入力式のEOSは通常パディング前にはないはずなので、単純に最初のEOSで区切る。
                        end_index = temp_end_index
                    except ValueError:
                        pass
                
                for token_id_idx in range(start_index, end_index):
                    token_id = raw_src_ids[token_id_idx]
                    if token_id == pad_id:
                        break
                    src_str_list.append(id_to_char.get(token_id, UNK_TOKEN))
                input_expression = "".join(src_str_list)

                target_str_list = []
                for token_id in current_tgt_tensor[:, 0].tolist():
                    if token_id == sos_id or token_id == eos_id or token_id == pad_id:
                        continue
                    target_str_list.append(id_to_char.get(token_id, UNK_TOKEN))
                correct_answer_str = "".join(target_str_list)

                predicted_str_list = []
                for token_id in generated_sequence_ids:
                    if token_id == sos_id : continue
                    if token_id == eos_id : break
                    if token_id == pad_id : break
                    predicted_str_list.append(id_to_char.get(token_id, UNK_TOKEN))
                model_prediction_str = "".join(predicted_str_list)
                
                is_correct = (model_prediction_str == correct_answer_str)
                if is_correct:
                    total_greedy_score += 1
                    correct_greedy_sequences +=1
                
                if analyze_results_filename: # ファイル名が指定されている場合のみリストに追加
                    analysis_results.append({
                        "Input Expression": input_expression,
                        "Correct Answer": correct_answer_str,
                        "Model Prediction": model_prediction_str,
                        "Is Correct": 1 if is_correct else 0
                    })
    
    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy_style_1 = total_greedy_score / 1000.0
    accuracy_style_2 = correct_greedy_sequences / total_sequences if total_sequences > 0 else 0
    
    if analyze_results_filename and analysis_results:
        try:
            df = pd.DataFrame(analysis_results)
            df.to_csv(analyze_results_filename, index=False, encoding='utf-8-sig')
            print(f"Prediction analysis saved to {analyze_results_filename}")
        except Exception as e:
            print(f"Error saving prediction analysis to {analyze_results_filename}: {e}")
            
    return avg_val_loss, total_greedy_score, accuracy_style_1, accuracy_style_2, total_sequences


if __name__ == '__main__':
    # --- 全ステージのデータを最初にロードして語彙を構築 ---
    all_curriculum_data_for_vocab = []
    for data_path in Config.CURRICULUM_DATA_PATHS:
        stage_data = load_csv_data(data_path)
        if stage_data:
            all_curriculum_data_for_vocab.append(stage_data)
    
    # 検証データと最終テストデータも語彙構築に含める
    val_data_for_vocab = load_csv_data(Config.VALIDATION_DATA_PATH)
    if val_data_for_vocab:
        all_curriculum_data_for_vocab.append(val_data_for_vocab)
    
    final_test_data_for_vocab = load_csv_data(Config.FINAL_TEST_DATA_PATH)
    if final_test_data_for_vocab:
        all_curriculum_data_for_vocab.append(final_test_data_for_vocab)

    if not all_curriculum_data_for_vocab:
        raise ValueError("No data found for vocabulary building. Please check CURRICULUM_DATA_PATHS and other data paths.")

    char_to_id, id_to_char = build_vocab(all_curriculum_data_for_vocab)
    vocab_size = len(char_to_id)
    PAD_ID = char_to_id[PAD_TOKEN]
    SOS_ID = char_to_id[SOS_TOKEN]
    EOS_ID = char_to_id[EOS_TOKEN]

    print(f"Vocabulary size (built from all stages): {vocab_size}")

    # --- 検証データローダーの準備 (全ステージ共通) ---
    val_data_pairs = load_csv_data(Config.VALIDATION_DATA_PATH)
    if not val_data_pairs:
        print("Warning: Validation data not found. Validation will be skipped.")
        val_dataloader = None # または空のデータローダー
    else:
        val_dataset = ArithmeticDataset(val_data_pairs, char_to_id)
        val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                                    collate_fn=lambda b: collate_fn(b, PAD_ID, DEVICE))

    # --- モデル、オプティマイザ、損失関数、スケジューラの初期化 ---
    model = Seq2SeqTransformer(input_vocab_size=vocab_size,
                               output_vocab_size=vocab_size,
                               d_model=Config.D_MODEL,
                               nhead=Config.NHEAD,
                               num_encoder_layers=Config.NUM_ENCODER_LAYERS,
                               num_decoder_layers=Config.NUM_DECODER_LAYERS,
                               dim_feedforward=Config.DIM_FEEDFORWARD,
                               dropout_p=Config.DROPOUT_P,
                               max_seq_len=Config.MAX_LEN).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), 
                           lr=Config.LEARNING_RATE,
                           betas=(Config.ADAM_BETA1, Config.ADAM_BETA2), 
                           eps=Config.ADAM_EPSILON)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=Config.LABEL_SMOOTHING)
    
    lr_scheduler = get_lr_scheduler(optimizer, Config.D_MODEL, Config.WARMUP_STEPS, 
                                    lr_scale_factor=Config.LR_SCALE_FACTOR)

    # --- 学習曲線記録用リストなど ---
    train_losses_overall = []
    val_losses_overall = []
    train_token_accuracies_overall = []
    val_seq_scores_overall = []
    val_seq_accuracies_overall = []
    learning_rates_overall = []

    best_val_score = -1
    epochs_no_improve = 0
    best_model_state_dict = None
    current_total_steps = 0
    overall_epoch_count = 0

    print(f"\nStarting Curriculum Training with Early Stopping (patience={Config.EARLY_STOPPING_PATIENCE}) and LR Scheduler (warmup_steps={Config.WARMUP_STEPS})...")

    # --- カリキュラムステージごとの学習ループ ---
    for stage_idx, train_data_path in enumerate(Config.CURRICULUM_DATA_PATHS):
        epochs_for_this_stage = Config.CURRICULUM_EPOCHS_PER_STAGE[stage_idx]
        print(f"\n--- Starting Curriculum Stage {stage_idx + 1}/{len(Config.CURRICULUM_DATA_PATHS)} ---")
        print(f"Training data: {train_data_path}, Epochs for this stage: {epochs_for_this_stage}")

        current_stage_train_data = load_csv_data(train_data_path)
        if not current_stage_train_data:
            print(f"Warning: No data found for stage {stage_idx + 1} at {train_data_path}. Skipping this stage.")
            continue
        
        train_dataset_stage = ArithmeticDataset(current_stage_train_data, char_to_id)
        train_dataloader_stage = DataLoader(train_dataset_stage, batch_size=Config.BATCH_SIZE, shuffle=True,
                                       collate_fn=lambda b: collate_fn(b, PAD_ID, DEVICE))
        
        if len(train_dataloader_stage) == 0:
            print(f"Warning: Training dataloader for stage {stage_idx + 1} is empty. Skipping this stage.")
            continue

        for epoch_in_stage in range(1, epochs_for_this_stage + 1):
            overall_epoch_count += 1
            print(f"\nOverall Epoch {overall_epoch_count} (Stage {stage_idx + 1}, Epoch in Stage {epoch_in_stage}/{epochs_for_this_stage})")

            train_loss, train_token_acc, current_total_steps, current_lr = train_epoch(
                model, train_dataloader_stage, optimizer, criterion, DEVICE, PAD_ID, lr_scheduler, current_total_steps
            )
            
            if val_dataloader: # 検証データがある場合のみ評価
                val_loss, val_score, val_acc_style1, val_acc_style2, _ = evaluate_model(
                    model, val_dataloader, criterion, DEVICE, id_to_char, char_to_id,
                    Config.MAX_LEN, SOS_ID, EOS_ID, PAD_ID,
                    analyze_results_filename=None # 検証時はCSV出力しない
                )
            else: # 検証データがない場合はダミー値を設定
                val_loss, val_score, val_acc_style1, val_acc_style2 = 0, 0, 0, 0
                print("  Validation skipped as no validation data is available.")

            
            train_losses_overall.append(train_loss)
            train_token_accuracies_overall.append(train_token_acc)
            val_losses_overall.append(val_loss)
            val_seq_scores_overall.append(val_score)
            val_seq_accuracies_overall.append(val_acc_style2)
            learning_rates_overall.append(current_lr)

            print(f"  LR: {current_lr:.8f}")
            print(f"  Train Loss: {train_loss:.4f}, Train Token Acc: {train_token_acc:.4f}")
            if val_dataloader:
                print(f"  Val Loss: {val_loss:.4f}, Val Seq Acc: {val_acc_style2:.4f} (Score: {val_score})")

            if val_dataloader and val_score > best_val_score: # Early stopping は検証スコアで判断
                best_val_score = val_score
                epochs_no_improve = 0
                best_model_state_dict = copy.deepcopy(model.state_dict())
                print(f"  Validation score improved to {val_score}. Saving model state.")
            elif val_dataloader:
                epochs_no_improve += 1
                print(f"  Validation score did not improve for {epochs_no_improve} epoch(s). Best score: {best_val_score}")

            if val_dataloader and epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after Overall Epoch {overall_epoch_count} due to no improvement for {Config.EARLY_STOPPING_PATIENCE} consecutive epochs.")
                break # 現在のステージのループを抜ける
        
        if val_dataloader and epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
            break # 全体の学習ループ (ステージループ) も抜ける

    print("\nCurriculum Training finished.")

    if best_model_state_dict:
        print("Loading best model state for final evaluation and saving.")
        model.load_state_dict(best_model_state_dict)
    else:
        print("No best model state was saved (e.g., validation was skipped or no improvement). Using last model state.")

    print(f"\nSaving final model to {Config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("Model saved.")

    log_header = ["overall_epoch", "train_loss", "train_token_acc", "val_loss", "val_seq_score", "val_seq_acc", "learning_rate"]
    log_data = []
    num_logged_epochs = len(train_losses_overall)
    for i in range(num_logged_epochs):
        log_data.append([
            i + 1, 
            train_losses_overall[i], train_token_accuracies_overall[i], 
            val_losses_overall[i], val_seq_scores_overall[i], 
            val_seq_accuracies_overall[i], learning_rates_overall[i]
        ])
    
    try:
        with open(Config.LOG_FILE_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
            writer.writerows(log_data)
        print(f"Training log saved to {Config.LOG_FILE_PATH}")
    except IOError:
        print(f"Error: Could not write training log to {Config.LOG_FILE_PATH}")

    # --- 最終テストデータでの評価 ---
    final_test_data_pairs = load_csv_data(Config.FINAL_TEST_DATA_PATH)
    if final_test_data_pairs:
        final_test_dataset = ArithmeticDataset(final_test_data_pairs, char_to_id)
        final_test_dataloader = DataLoader(final_test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                                     collate_fn=lambda b: collate_fn(b, PAD_ID, DEVICE))
        
        if final_test_dataloader and len(final_test_dataloader) > 0:
            print("\n--- Final Evaluation on Dedicated Test Set (with best model) ---")
            test_loss, test_total_score, test_accuracy_style_1, test_accuracy_style_2, num_test_samples = evaluate_model(
                model, final_test_dataloader, criterion, DEVICE, id_to_char, char_to_id,
                Config.MAX_LEN, SOS_ID, EOS_ID, PAD_ID,
                analyze_results_filename=Config.PREDICTION_ANALYSIS_FILENAME
            )
            print(f"Test Set: Total Samples: {num_test_samples}")
            print(f"Test Set: Loss: {test_loss:.4f}")
            print(f"Test Set: Total Score (Correct Sequences): {test_total_score}")
            print(f"Test Set: Accuracy (Total Score / 1000): {test_accuracy_style_1:.4f}")
            print(f"Test Set: Accuracy (Correct Sequences / Total Samples): {test_accuracy_style_2:.4f}")
        else:
            print("\nFinal test dataloader is empty or could not be created. Skipping final test evaluation.")
    else:
        print("\nNo final test data loaded. Skipping final test evaluation.")
    
    print("\n--- Inference Example (with best model) ---")
    
    model.eval()
    with torch.no_grad():
        test_eq = "25+30=" 
        src_tensor = torch.tensor(tokenize_and_numericalize(test_eq, char_to_id), dtype=torch.long).unsqueeze(1).to(DEVICE)
        src_padding_mask_infer = (src_tensor == PAD_ID).transpose(0,1).to(DEVICE)

        memory_infer = model.encode(src_tensor, src_mask=None, src_padding_mask=src_padding_mask_infer)
        # current_memory_key_padding_mask_infer = src_padding_mask_infer # This var is not used later in decode

        generated_ids_infer = [SOS_ID]
        ys_infer = torch.ones(1, 1).fill_(SOS_ID).type_as(src_tensor.data).to(DEVICE)

        for _ in range(Config.MAX_LEN -1):
            tgt_causal_mask_infer = generate_square_subsequent_mask(ys_infer.size(0), DEVICE)
            current_tgt_padding_mask_infer = torch.zeros(ys_infer.size(1), ys_infer.size(0), dtype=torch.bool, device=DEVICE)
            
            # memory_key_padding_mask はエンコーダーのパディングマスク (src_padding_mask_infer) を渡す
            decoder_output_infer = model.decode(ys_infer, memory_infer[:,0:1,:], tgt_causal_mask_infer,
                                                current_tgt_padding_mask_infer, src_padding_mask_infer[0:1,:]) # 修正: memory_key_padding_mask
            
            last_token_logits_infer = model.generator(decoder_output_infer[-1, :, :])
            next_token_id_infer = last_token_logits_infer.argmax(dim=-1).item()
            generated_ids_infer.append(next_token_id_infer)

            if next_token_id_infer == EOS_ID:
                break
            
            new_token_tensor_infer = torch.ones(1, 1).type_as(src_tensor.data).fill_(next_token_id_infer).to(DEVICE)
            ys_infer = torch.cat((ys_infer, new_token_tensor_infer), dim=0)
        
        result_chars = []
        for gid in generated_ids_infer:
            if gid == SOS_ID: continue
            if gid == EOS_ID: break
            result_chars.append(id_to_char.get(gid, UNK_TOKEN))
        
        print(f"Input: {test_eq}")
        print(f"Predicted: {''.join(result_chars)}")
