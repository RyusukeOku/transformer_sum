import random
import csv
from typing import Set, Tuple, List, Optional

def generate_addition_problems(
    num_samples: int,
    num1_digits: int,
    num2_digits: int,
    allow_carry: Optional[bool] = None, # None: 気にしない, True: 繰り上がりのみを許容, False: 繰り上がりなしのみを許容
    force_result_digits: Optional[int] = None, # 結果の桁数を強制する場合
    existing_samples: Optional[Set[Tuple[str, str]]] = None
) -> Set[Tuple[str, str]]:
    """
    指定された条件に基づいて足し算の問題と答えのペアを生成する。

    Args:
        num_samples (int): 生成するターゲットのサンプル数。
        num1_digits (int): 最初の数の桁数。
        num2_digits (int): 2番目の数の桁数。
        allow_carry (Optional[bool]): 繰り上がりを許容するかどうか。
                                     Noneなら不問、Trueなら繰り上がりのあるもの、Falseなら繰り上がりのないもの。
        force_result_digits (Optional[int]): 結果の桁数を指定する場合。
        existing_samples (Optional[Set[Tuple[str, str]]]): 既に存在するサンプルのセット。

    Returns:
        Set[Tuple[str, str]]: (式文字列, 答え文字列) のタプルのセット。
    """
    if existing_samples is None:
        existing_samples = set()
    
    new_samples: Set[Tuple[str, str]] = set()
    
    # 試行回数の上限を設定 (無限ループ防止)
    # 組み合わせが少ない条件では、要求されたサンプル数を生成できない場合がある
    max_attempts = num_samples * 50  # 条件によっては組み合わせが少ないため、試行回数を多めに
    attempts = 0

    min_num1 = 10**(num1_digits - 1) if num1_digits > 0 else 0
    max_num1 = (10**num1_digits) - 1 if num1_digits > 0 else 0
    min_num2 = 10**(num2_digits - 1) if num2_digits > 0 else 0
    max_num2 = (10**num2_digits) - 1 if num2_digits > 0 else 0
    
    if num1_digits == 1 and min_num1 == 1: min_num1 = 0 # 1桁の場合は0も含むように調整
    if num2_digits == 1 and min_num2 == 1: min_num2 = 0

    while len(new_samples) < num_samples and attempts < max_attempts:
        attempts += 1
        num1 = random.randint(min_num1, max_num1)
        num2 = random.randint(min_num2, max_num2)
        
        result = num1 + num2
        expression = f"{num1}+{num2}="
        answer = str(result)
        
        # --- 条件チェック ---
        # 1. 繰り上がり条件
        if allow_carry is not None:
            # 簡易的な繰り上がりチェック (各桁の和が9を超えるか)
            # より厳密には、筆算のプロセスをシミュレートする必要があるが、ここでは簡易版
            s_num1, s_num2 = str(num1), str(num2)
            max_len = max(len(s_num1), len(s_num2))
            s_num1 = s_num1.zfill(max_len)
            s_num2 = s_num2.zfill(max_len)
            has_carry_in_op = False
            for i in range(max_len):
                if int(s_num1[i]) + int(s_num2[i]) >= 10: # 1の位から見ていくなら逆順で
                    has_carry_in_op = True
                    break
            
            # 答えの桁数とオペランドの最大桁数を比較することでも簡易的に判定可能
            # (ただし、9+1=10のようなケースも繰り上がりとみなすか、9+0=9のようなケースをどう扱うかによる)
            # ここでは、答えの桁数がオペランドの最大桁数より大きい場合、または
            # 各桁の和で10以上になる箇所があれば繰り上がりとみなす
            result_has_more_digits = len(answer) > max(len(str(num1)), len(str(num2)))

            # 厳密な「繰り上がりなし」は、各桁の和が9以下であること
            is_no_carry_strict = True
            temp_n1, temp_n2 = num1, num2
            while temp_n1 > 0 or temp_n2 > 0:
                digit_sum = (temp_n1 % 10) + (temp_n2 % 10)
                if digit_sum >= 10:
                    is_no_carry_strict = False
                    break
                temp_n1 //= 10
                temp_n2 //= 10

            current_has_carry = not is_no_carry_strict # 厳密な繰り上がりなしの否定 ＝ 繰り上がりあり

            if allow_carry is True and not current_has_carry: # 繰り上がり必須なのに、繰り上がりがない
                continue
            if allow_carry is False and current_has_carry: # 繰り上がりなし必須なのに、繰り上がりがある
                continue
        
        # 2. 結果の桁数条件
        if force_result_digits is not None:
            if len(answer) != force_result_digits:
                continue
        
        sample_pair = (expression, answer)
        
        if sample_pair not in existing_samples and sample_pair not in new_samples:
            new_samples.add(sample_pair)
            
    if len(new_samples) < num_samples:
        print(f"Warning (Stage: {num1_digits}d+{num2_digits}d, Carry:{allow_carry}, ResDigits:{force_result_digits}): "
              f"Could only generate {len(new_samples)} unique samples out of {num_samples} requested.")
              
    return new_samples

def save_dataset_to_csv(dataset: Set[Tuple[str, str]], filename: str, shuffle: bool = True):
    dataset_list = list(dataset)
    if shuffle:
        random.shuffle(dataset_list)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for expression_str, answer_str in dataset_list:
            writer.writerow([expression_str, answer_str])
    print(f"Dataset saved to {filename} ({len(dataset_list)} samples). Shuffled: {shuffle}")

if __name__ == "__main__":
    # --- カリキュラムステージごとのデータ生成 ---
    # 注意: 各ステージで生成されるサンプル数は、条件によって組み合わせの総数が限られるため、
    # 要求した数に満たない場合があります。その場合はWarningが出力されます。

    all_generated_samples: Set[Tuple[str, str]] = set() # 全ステージでの重複を避けるため

    # ステージ1: 1桁 + 1桁 (繰り上がりなし)
    s1_samples = 50 # 例: 0+0=0 から 4+5=9 までなど (全45通りのはず)
    stage1_data = generate_addition_problems(s1_samples, 1, 1, allow_carry=False, existing_samples=all_generated_samples)
    all_generated_samples.update(stage1_data)
    save_dataset_to_csv(stage1_data, "data/curriculum_stage1_1d_1d_nocarry.csv")

    # ステージ2: 1桁 + 1桁 (繰り上がりあり)
    s2_samples = 50 # 例: 1+9=10 から 9+9=18 までなど (36通りのはず)
    stage2_data = generate_addition_problems(s2_samples, 1, 1, allow_carry=True, existing_samples=all_generated_samples)
    all_generated_samples.update(stage2_data)
    save_dataset_to_csv(stage2_data, "data/curriculum_stage2_1d_1d_carry.csv")

    # ステージ3: 2桁 + 1桁 (繰り上がりなし)
    s3_samples = 500
    stage3_data = generate_addition_problems(s3_samples, 2, 1, allow_carry=False, existing_samples=all_generated_samples)
    all_generated_samples.update(stage3_data)
    save_dataset_to_csv(stage3_data, "data/curriculum_stage3_2d_1d_nocarry.csv")

    # ステージ4: 2桁 + 1桁 (繰り上がりあり)
    s4_samples = 500
    stage4_data = generate_addition_problems(s4_samples, 2, 1, allow_carry=True, existing_samples=all_generated_samples)
    all_generated_samples.update(stage4_data)
    save_dataset_to_csv(stage4_data, "data/curriculum_stage4_2d_1d_carry.csv")
    
    # ステージ5: 2桁 + 2桁 (繰り上がりなし、または1の位のみで繰り上がり、答えが2桁)
    # この条件は複雑なので、より単純に「答えが2桁」でフィルタリングし、allow_carryはNoneにするか、
    # allow_carry=False (厳密な繰り上がりなし) を試す。
    s5_samples = 2000
    # 例1: 答えが2桁の2d+2d
    stage5_data_res2 = generate_addition_problems(s5_samples, 2, 2, force_result_digits=2, existing_samples=all_generated_samples)
    all_generated_samples.update(stage5_data_res2)
    save_dataset_to_csv(stage5_data_res2, "data/curriculum_stage5a_2d_2d_res2digit.csv")
    # 例2: 厳密に繰り上がりがない2d+2d (この場合、答えは必ず2桁になるはず)
    # stage5_data_nocarry = generate_addition_problems(s5_samples, 2, 2, allow_carry=False, existing_samples=all_generated_samples)
    # all_generated_samples.update(stage5_data_nocarry)
    # save_dataset_to_csv(stage5_data_nocarry, "curriculum_stage5b_2d_2d_nocarry.csv")


    # ステージ6: 2桁 + 2桁 (10の位でも繰り上がりが発生し、答えが3桁になるもの)
    s6_samples = 4000 # 2d+2d で答えが3桁になるのは、10+90=100 のようなケースから99+99=198まで
    stage6_data = generate_addition_problems(s6_samples, 2, 2, force_result_digits=3, existing_samples=all_generated_samples)
    all_generated_samples.update(stage6_data)
    save_dataset_to_csv(stage6_data, "data/curriculum_stage6_2d_2d_res3digit_carry.csv")

    # オプション: 全ての生成データを結合して、一つの大きなデータセットとして保存も可能
    # combined_filename = "addition_curriculum_combined_dataset.csv"
    # save_dataset_to_csv(all_generated_samples, combined_filename, shuffle=True)
    # print(f"\nTotal unique samples generated across all stages: {len(all_generated_samples)}")
