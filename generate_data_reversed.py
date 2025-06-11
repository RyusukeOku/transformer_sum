import random
import csv

def generate_unique_addition_samples(num_samples: int, existing_samples: set = None):
    """
    指定された数の、2桁の数字同士の足し算の式と答えのペアを生成する。
    A+B= と B+A= の両方の形式が含まれるように試みる。
    式は標準形式（例: "25+30="）とする。
    existing_samples に含まれるデータは生成しない。

    Args:
        num_samples (int): 生成するターゲットのサンプル数。
        existing_samples (set, optional): 既に存在するサンプルのセット。
                                         このセットに含まれるサンプルは生成対象外となる。
                                         デフォルトは None (既存サンプルなし)。

    Returns:
        set: (式文字列, 答え文字列) のタプルのセット。
    """
    if existing_samples is None:
        existing_samples = set()
    
    new_samples = set()
    
    max_attempts = num_samples * 25  # 十分な試行回数を設定
    attempts = 0

    while len(new_samples) < num_samples and attempts < max_attempts:
        num1 = random.randint(10, 99)  # 2桁の数字
        num2 = random.randint(10, 99)  # 2桁の数字
        
        answer = str(num1 + num2)

        # 形式1: num1 + num2
        expression1 = f"{num1}+{num2}="
        sample_pair1 = (expression1, answer)
        
        added_in_iteration = False
        if sample_pair1 not in existing_samples and sample_pair1 not in new_samples:
            if len(new_samples) < num_samples: # ターゲット数に達していなければ追加
                new_samples.add(sample_pair1)
                added_in_iteration = True

        # 形式2: num2 + num1 (num1 と num2 が異なる場合)
        if num1 != num2 and len(new_samples) < num_samples: # ターゲット数に達していなければ追加試行
            expression2 = f"{num2}+{num1}="
            sample_pair2 = (expression2, answer)
            if sample_pair2 not in existing_samples and sample_pair2 not in new_samples:
                new_samples.add(sample_pair2)
                added_in_iteration = True
        
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
            writer.writerow([expression_str, answer_str])
            
    print(f"Dataset saved to {filename} ({len(dataset_list)} samples). Shuffled: {shuffle}")


if __name__ == "__main__":
    # 2桁+2桁のユニークな式の組み合わせは約8100通りなので、
    # 訓練データとテストデータの合計がこれを超えないように注意する。
    num_train_samples = 8000
    num_test_samples = 2000
    
    # 出力ファイル名を標準形式のものに戻す（または新しい名前を付ける）
    train_filename = "addition_train_commutative_dataset.csv"
    test_filename = "addition_test_commutative_dataset.csv"

    # 1. 訓練データセットの生成
    print(f"Generating training dataset ({num_train_samples} samples)...")
    train_dataset = generate_unique_addition_samples(num_train_samples)
    save_dataset_to_csv(train_dataset, train_filename, shuffle=True)
    
    # 2. テストデータセットの生成（訓練データと重複しないように）
    print(f"\nGenerating test dataset ({num_test_samples} samples)...")
    test_dataset = generate_unique_addition_samples(num_test_samples, existing_samples=train_dataset)
    save_dataset_to_csv(test_dataset, test_filename, shuffle=True)

    if train_dataset and test_dataset:
        intersection_count = len(train_dataset.intersection(test_dataset))
        if intersection_count == 0:
            print(f"\nSuccessfully generated train and test datasets with no overlap.")
        else:
            print(f"\nError: Found {intersection_count} overlapping samples between train and test datasets.")
    elif not train_dataset:
        print("\nWarning: Training dataset is empty.")
    elif not test_dataset:
        print("\nWarning: Test dataset is empty.")