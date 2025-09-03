from preprocessing import Preprocessing
from modeling import train_xgb, train_global
from ensemble import weighted_ensemble
from postprocessing import apply_holiday_adjustment
from config import OUTPUT_DIR
import os


def run_type_model(train, test, summer_flag):
    # building_type 모델 학습
    X = train.drop(['solar_power_capacity','ess_capacity','pcs_capacity',
                    'power_consumption','rainfall','sunshine','solar_radiation',
                    'hour','day','month','day_of_week','date_time'], axis=1)
    Y = train[['building_type','power_consumption']]
    test_X = test.drop(['solar_power_capacity','ess_capacity','pcs_capacity',
                        'rainfall','hour','month','day_of_week','day','date_time'], axis=1)

    max_depth_dict = {
        'Other Buildings': 10, 'Public': 10, 'University': 8, 'IDC': 6,
        'Department Store': 8, 'Hospital': 8, 'Commercial': 10,
        'Apartment': 6, 'Research Institute': 10, 'Hotel': 10
    }

    answer, pred = train_xgb(X, Y, test_X, group_col="building_type", max_depth_dict=max_depth_dict)
    answer.to_csv(f"{OUTPUT_DIR}/answer_type_summer{summer_flag}.csv", index=False)


def run_number_model(train, test, summer_flag):
    # building_number 모델 학습
    X = train.drop(['solar_power_capacity','ess_capacity','pcs_capacity',
                    'power_consumption','rainfall','sunshine','solar_radiation',
                    'hour','day','month','day_of_week','date_time'], axis=1)
    Y = train[['building_number','power_consumption']]
    test_X = test.drop(['solar_power_capacity','ess_capacity','pcs_capacity',
                        'rainfall','hour','month','day_of_week','day','date_time'], axis=1)

    # building_type 기반 max_depth 매핑
    btype_map = dict(zip(train['building_number'], train['building_type']))
    max_depth_dict = {b: 10 for b in train['building_number'].unique()}
    for b, bt in btype_map.items():
        if bt == 'University': max_depth_dict[b] = 8
        elif bt == 'IDC': max_depth_dict[b] = 6
        elif bt == 'Department Store': max_depth_dict[b] = 8
        elif bt == 'Hospital': max_depth_dict[b] = 8
        elif bt == 'Apartment': max_depth_dict[b] = 6

    answer, pred = train_xgb(X, Y, test_X, group_col="building_number", max_depth_dict=max_depth_dict)
    answer.to_csv(f"{OUTPUT_DIR}/answer_number_summer{summer_flag}.csv", index=False)


def run_cluster_model(train, test, summer_flag):
    # cluster 모델 학습
    X = train.drop(['solar_power_capacity','ess_capacity','pcs_capacity',
                    'power_consumption','rainfall','sunshine','solar_radiation',
                    'hour','day','month','day_of_week','date_time','building_type'], axis=1)
    Y = train[['cluster','power_consumption']]
    test_X = test.drop(['solar_power_capacity','ess_capacity','pcs_capacity',
                        'rainfall','hour','month','day_of_week','day','date_time'], axis=1)

    max_depth_dict = {0:10, 1:8, 2:10, 3:8, 4:10}
    answer, pred = train_xgb(X, Y, test_X, group_col="cluster", max_depth_dict=max_depth_dict)
    answer.to_csv(f"{OUTPUT_DIR}/answer_cluster_summer{summer_flag}.csv", index=False)


def run_global_model(train, test, summer_flag):
    # global 모델 학습
    X = pd.get_dummies(train.drop(['solar_power_capacity','ess_capacity','pcs_capacity',
                                   'power_consumption','rainfall','sunshine','solar_radiation',
                                   'hour','day','month','day_of_week','date_time'], axis=1),
                       columns=["building_type","building_number"], drop_first=False)

    test_X = pd.get_dummies(test.drop(['solar_power_capacity','ess_capacity','pcs_capacity',
                                       'rainfall','hour','month','day_of_week','day','date_time'], axis=1),
                            columns=["building_type","building_number"], drop_first=False)

    test_X = test_X.reindex(columns=X.columns, fill_value=0)
    Y = train[['power_consumption']].copy()

    answer, pred = train_global(X, Y, test_X, max_depth=10)
    answer.to_csv(f"{OUTPUT_DIR}/answer_global_summer{summer_flag}.csv", index=False)


def main():
    # --------------------
    # 1. 8개 모델 학습
    # --------------------
    for summer_flag in [0, 1]:
        # type
        train, test = Preprocessing(summer=bool(summer_flag), cluster=False)
        run_type_model(train, test, summer_flag)

        # number
        train, test = Preprocessing(summer=bool(summer_flag), cluster=False)
        run_number_model(train, test, summer_flag)

        # cluster
        train, test = Preprocessing(summer=bool(summer_flag), cluster=True)
        run_cluster_model(train, test, summer_flag)

        # global
        train, test = Preprocessing(summer=bool(summer_flag), cluster=False)
        run_global_model(train, test, summer_flag)

    # --------------------
    # 2. 앙상블
    # --------------------
    weighted_ensemble()

    # --------------------
    # 3. 후처리
    # --------------------
    apply_holiday_adjustment()


if __name__ == "__main__":
    import pandas as pd
    main()