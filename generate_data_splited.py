import random
import csv

def generate_unique_addition_samples(num_samples: int, existing_samples: set = None):
    """
    指定された数の、2桁の数字同士の足し算の式と答えのペアを生成する。
    式は (10の位+1の位)+(10の位+1の位)= の形式に分解される。
    existing_samples に含まれるデータは生成しない。

    Args:
        num_samples (int): 生成するサンプル数。
        existing_samples (set, optional): 既に存在するサンプルのセット。
                                         このセットに含まれるサンプルは生成対象外となる。
                                         デフォルトは None (既存サンプルなし)。

    Returns:
        set: (分解された式文字列, 答え文字列) のタプルのセット。
    """
    if existing_samples is None:
        existing_samples = set()
    
    new_samples = set()
    
    # 2桁+2桁の全組み合わせ数は約8100通り。
    # 要求サンプル数が多すぎる場合の警告のため、試行回数に上限を設ける。
    max_attempts = num_samples * 25 # 十分な試行回数を設定 (以前より少し増やしました)
    attempts = 0

    while len(new_samples) < num_samples and attempts < max_attempts:
        num1 = random.randint(10, 99) # 2桁の数字
        num2 = random.randint(10, 99) # 2桁の数字
        
        # num1 を分解
        num1_tens = (num1 // 10) * 10
        num1_units = num1 % 10
        
        # num2 を分解
        num2_tens = (num2 // 10) * 10
        num2_units = num2 % 10
        
        # 分解された形式の式文字列を生成
        # 例: 51 + 19 =  -> (50+1)+(10+9)=
        expression = f"({num1_tens}+{num1_units})+({num2_tens}+{num2_units})="
        answer = str(num1 + num2) # 答えは通常の計算結果
        
        sample_pair = (expression, answer)
        
        # 既存のサンプルや、今回既に生成したサンプルと重複しない場合のみ追加
        if sample_pair not in existing_samples and sample_pair not in new_samples:
            new_samples.add(sample_pair)
        
        attempts += 1
        
    if len(new_samples) < num_samples:
        print(f"Warning: Could only generate {len(new_samples)} unique samples out of the requested {num_samples}. "
              f"This can happen if the requested number of unique samples is close to or exceeds the "
              f"total number of unique possible combinations (~8100 for 2-digit+2-digit), "
              f"especially when excluding already existing samples.")
              
    return new_samples

def save_dataset_to_csv(dataset: set, filename: str, shuffle: bool = False):
    """
    データセットをCSVファイルとして保存する。オプションでシャッフルする。

    Args:
        dataset (set): (式文字列, 答え文字列) のタプルのセット。
        filename (str): 保存先のファイル名。
        shuffle (bool): Trueの場合、書き出す前にデータセットの行をシャッフルする。
    """
    
    dataset_list = list(dataset)
    
    if shuffle:
        random.shuffle(dataset_list)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for expression_str, answer_str in dataset_list:
            writer.writerow([expression_str, answer_str]) # 各ペアを1行として書き込む
            
    print(f"Dataset saved to {filename} ({len(dataset_list)} samples). Shuffled: {shuffle}")


if __name__ == "__main__":
    # 2桁+2桁のユニークな式の組み合わせは約8100通りなので、
    # 訓練データとテストデータの合計がこれを超えないように注意する。
    # (超えた場合、generate_unique_addition_samples関数内で警告が表示され、
    #  可能な限りのユニークなサンプルが生成されます)
    num_train_samples = 8000  # 訓練データのサンプル数を指定
    num_test_samples = 2000   # テストデータのサンプル数を指定 (合計7500)
    
    train_filename = "addition_train_decomposed_dataset.csv"
    test_filename = "addition_test_decomposed_dataset.csv"

    # 1. 訓練データセットの生成
    print(f"Generating training dataset ({num_train_samples} samples)...")
    train_dataset = generate_unique_addition_samples(num_train_samples)
    save_dataset_to_csv(train_dataset, train_filename, shuffle=True)
    
    # 2. テストデータセットの生成（訓練データと重複しないように）
    print(f"\nGenerating test dataset ({num_test_samples} samples)...")
    # 訓練データを既存サンプルとして渡す
    test_dataset = generate_unique_addition_samples(num_test_samples, existing_samples=train_dataset)
    save_dataset_to_csv(test_dataset, test_filename, shuffle=True)

    # 生成されたデータセットの重複確認 (オプション)
    if train_dataset and test_dataset: # 両方にデータがある場合のみチェック
        intersection_count = len(train_dataset.intersection(test_dataset))
        if intersection_count == 0:
            print(f"\nSuccessfully generated train and test datasets with no overlap.")
        else:
            print(f"\nError: Found {intersection_count} overlapping samples between train and test datasets.")
    elif not train_dataset:
        print("\nWarning: Training dataset is empty.")
    elif not test_dataset:
        print("\nWarning: Test dataset is empty (possibly all unique combinations were used for training).")