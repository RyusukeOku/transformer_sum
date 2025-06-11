import random
import csv # csvモジュールをインポート (より堅牢なCSV書き出しのため)

def generate_unique_addition_samples(num_samples: int, existing_samples: set = None):
    """
    指定された数の、2桁の数字同士の足し算の式と答えのペアを生成する。
    existing_samples に含まれるデータは生成しない。

    Args:
        num_samples (int): 生成するサンプル数。
        existing_samples (set, optional): 既に存在するサンプルのセット。
                                         このセットに含まれるサンプルは生成対象外となる。
                                         デフォルトは None (既存サンプルなし)。

    Returns:
        set: (式文字列, 答え文字列) のタプルのセット。
    """
    if existing_samples is None:
        existing_samples = set()
    
    new_samples = set()
    
    max_attempts = num_samples * 20
    attempts = 0

    while len(new_samples) < num_samples and attempts < max_attempts:
        num1 = random.randint(100, 999)
        num2 = random.randint(100, 999)
        
        expression = f"{num1}+{num2}="
        answer = str(num1 + num2)
        
        sample_pair = (expression, answer)
        
        if sample_pair not in existing_samples and sample_pair not in new_samples:
            new_samples.add(sample_pair)
        
        attempts += 1
        
    if len(new_samples) < num_samples:
        print(f"Warning: Could only generate {len(new_samples)} unique samples out of the requested {num_samples} "
              f"that are not in existing_samples. This might happen if the requested number "
              f"is too large compared to the possible unique combinations minus existing samples.")
              
    return new_samples

def save_dataset_to_csv(dataset: set, filename: str, shuffle: bool = False): # shuffle引数を追加
    """
    データセットをCSVファイルとして保存する。オプションでシャッフルする。

    Args:
        dataset (set): (式文字列, 答え文字列) のタプルのセット。
        filename (str): 保存先のファイル名。
        shuffle (bool): Trueの場合、書き出す前にデータセットの行をシャッフルする。
                        デフォルトは False。
    """
    
    dataset_list = list(dataset) # まずリストに変換
    
    if shuffle:
        random.shuffle(dataset_list) # リストをシャッフル
    else:
        # 以前はソートしていたが、シャッフルしない場合はセットからの変換順 (不定) のままか、
        # あるいは一貫性のためにソートするかを選択できる。ここではソートしない。
        # もし以前の挙動（ソート）が良い場合は以下を有効化:
        # dataset_list.sort() 
        pass

    with open(filename, 'w', newline='', encoding='utf-8') as f: # encodingを指定
        writer = csv.writer(f) # csv.writerを使用
        # ヘッダーは不要なので書き込まない
        for expression_str, answer_str in dataset_list:
            writer.writerow([expression_str, answer_str]) # writerowで行を書き込む
            
    print(f"Dataset saved to {filename} ({len(dataset_list)} samples). Shuffled: {shuffle}")


if __name__ == "__main__":
    num_train_samples = 8000
    num_test_samples = 2000
    
    train_filename = "data/addition_train_dataset_3d.csv"
    test_filename = "data/addition_test_dataset_3d.csv"

    # 1. 訓練データセットの生成
    print(f"Generating training dataset ({num_train_samples} samples)...")
    train_dataset = generate_unique_addition_samples(num_train_samples)
    # 訓練データをシャッフルして保存
    save_dataset_to_csv(train_dataset, train_filename, shuffle=True) 
    
    # 2. テストデータセットの生成（訓練データと重複しないように）
    print(f"\nGenerating test dataset ({num_test_samples} samples)...")
    test_dataset = generate_unique_addition_samples(num_test_samples, existing_samples=train_dataset)
    # テストデータもシャッフルして保存 (通常、テストデータの順序は評価に影響しないが、一貫性のためにシャッフルしても良い)
    save_dataset_to_csv(test_dataset, test_filename, shuffle=True)

    intersection_count = len(train_dataset.intersection(test_dataset))
    if intersection_count == 0:
        print(f"\nSuccessfully generated train and test datasets with no overlap.")
    else:
        print(f"\nError: Found {intersection_count} overlapping samples between train and test datasets.")