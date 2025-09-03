import pandas as pd
from config import OUTPUT_DIR

def load_and_sort(path):
    df = pd.read_csv(path)
    return df.sort_index()["answer"].values

def weighted_ensemble():
    """
    최종 앙상블 로직:
    [(summer의 type, number, global 각각 0.2/0.3/0.5) * 0.8 +
     (nosummer의 type, number, global 각각 0.25/0.25/0.5) * 0.2] * 0.8 +
     (summer cluster * 0.7 + nosummer cluster * 0.3)
    """

    # --------------------
    # Summer 결과 로드
    # --------------------
    summer_type   = load_and_sort(f"{OUTPUT_DIR}/answer_type_summer1.csv")
    summer_number = load_and_sort(f"{OUTPUT_DIR}/answer_number_summer1.csv")
    summer_global = load_and_sort(f"{OUTPUT_DIR}/answer_global_summer1.csv")
    summer_cluster= load_and_sort(f"{OUTPUT_DIR}/answer_cluster_summer1.csv")

    # --------------------
    # NoSummer 결과 로드
    # --------------------
    nosummer_type   = load_and_sort(f"{OUTPUT_DIR}/answer_type_summer0.csv")
    nosummer_number = load_and_sort(f"{OUTPUT_DIR}/answer_number_summer0.csv")
    nosummer_global = load_and_sort(f"{OUTPUT_DIR}/answer_global_summer0.csv")
    nosummer_cluster= load_and_sort(f"{OUTPUT_DIR}/answer_cluster_summer0.csv")

    # --------------------
    # 가중치 조합
    # --------------------
    summer_combo = summer_type * 0.2 + summer_number * 0.3 + summer_global * 0.5
    nosummer_combo = nosummer_type * 0.25 + nosummer_number * 0.25 + nosummer_global * 0.5

    base_ensemble = (summer_combo * 0.8 + nosummer_combo * 0.2) * 0.8
    cluster_ensemble = summer_cluster * 0.7 + nosummer_cluster * 0.3

    final_pred = base_ensemble + cluster_ensemble

    # 음수 값 방지
    final_pred_fixed = [max(0, x) for x in final_pred]

    # --------------------
    # sample_submission 기반 최종 제출 파일 생성
    # --------------------
    submission = pd.read_csv(f"{OUTPUT_DIR}/../data/sample_submission.csv")
    submission['answer'] = final_pred_fixed
    submission.to_csv(f"{OUTPUT_DIR}/final_ensemble.csv", index=False)

    print(f"✔ Final ensemble saved to {OUTPUT_DIR}/final_ensemble.csv")
