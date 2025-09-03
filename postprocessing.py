import pandas as pd
import numpy as np
from config import DATA_DIR, OUTPUT_DIR

def apply_holiday_adjustment():
    """
    원본 holiday 후처리 로직 그대로 구현
    """
    train = pd.read_csv(f'{DATA_DIR}/train.csv', encoding='utf-8-sig')
    building_info = pd.read_csv(f'{DATA_DIR}/building_info.csv', encoding='utf-8-sig')

    # rename
    train = train.rename(columns={
        '건물번호': 'building_number',
        '일시': 'date_time',
        '기온(°C)': 'temperature',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed',
        '습도(%)': 'humidity',
        '일조(hr)': 'sunshine',
        '일사(MJ/m2)': 'solar_radiation',
        '전력소비량(kWh)': 'power_consumption'
    })
    train.drop('num_date_time', axis=1, inplace=True)

    building_info = building_info.rename(columns={
        '건물번호': 'building_number',
        '건물유형': 'building_type',
        '연면적(m2)': 'total_area',
        '냉방면적(m2)': 'cooling_area',
        '태양광용량(kW)': 'solar_power_capacity',
        'ESS저장용량(kWh)': 'ess_capacity',
        'PCS용량(kW)': 'pcs_capacity'
    })

    translation_dict = {
        '건물기타': 'Other Buildings', '공공': 'Public', '학교': 'University',
        '백화점': 'Department Store', '병원': 'Hospital', '상용': 'Commercial',
        '아파트': 'Apartment', '연구소': 'Research Institute',
        'IDC(전화국)': 'IDC', '호텔': 'Hotel'
    }
    building_info['building_type'] = building_info['building_type'].replace(translation_dict)

    train = pd.merge(train, building_info, on='building_number', how='left')
    train['date_time'] = pd.to_datetime(train['date_time'], format='%Y%m%d %H')
    train['hour'] = train['date_time'].dt.hour
    train['day'] = train['date_time'].dt.day
    train['month'] = train['date_time'].dt.month

    # outlier 제거
    outlier_idx = train.index[train['power_consumption'] == 0].tolist()
    train.drop(index=outlier_idx, inplace=True)

    # 앙상블 결과 로드
    submission_file_path = f'{OUTPUT_DIR}/final_ensemble.csv'
    submit = pd.read_csv(submission_file_path)

    test_raw = pd.read_csv(f'{DATA_DIR}/test.csv', encoding='utf-8-sig')
    test_raw = test_raw.rename(columns={'건물번호': 'building_number', '일시': 'date_time'})
    test_raw['date_time'] = pd.to_datetime(test_raw['date_time'], format='%Y%m%d %H')
    test_raw['hour'] = test_raw['date_time'].dt.hour
    test_raw['day'] = test_raw['date_time'].dt.day
    test_raw['month'] = test_raw['date_time'].dt.month

    submit = submit.merge(test_raw[['building_number', 'date_time', 'hour', 'day', 'month']],
                          left_index=True, right_index=True, how='left')

    # replacement rules (원본과 동일)
    replacement_rules = {
        29: {'target_date': {'month': 8, 'day': 25}, 'source_dates': [{'month': 6, 'days': [23]}, {'month': 7, 'days': [28]}]},
        27: {'target_date': {'month': 8, 'day': 25}, 'source_dates': [{'month': 6, 'days': [9, 23]}, {'month': 7, 'days': [14, 28]}, {'month': 8, 'days': [11]}]},
        32: {'target_date': {'month': 8, 'day': 26}, 'source_dates': [{'month': 6, 'days': [10, 24]}, {'month': 7, 'days': [8, 22]}, {'month': 8, 'days': [12]}]},
        40: {'target_date': {'month': 8, 'day': 25}, 'source_dates': [{'month': 6, 'days': [9, 23]}, {'month': 7, 'days': [14, 28]}, {'month': 8, 'days': [11]}]},
        59: {'target_date': {'month': 8, 'day': 25}, 'source_dates': [{'month': 6, 'days': [9, 23]}, {'month': 7, 'days': [14, 28]}, {'month': 8, 'days': [11]}]},
        63: {'target_date': {'month': 8, 'day': 25}, 'source_dates': [{'month': 6, 'days': [9, 23]}, {'month': 7, 'days': [14, 28]}, {'month': 8, 'days': [11]}]}
    }

    for building_num, rules in replacement_rules.items():
        target_month = rules['target_date']['month']
        target_day = rules['target_date']['day']
        source_dates = rules['source_dates']

        for hour in range(24):
            target_indices = submit[(submit['building_number'] == building_num) &
                                    (submit['month'] == target_month) &
                                    (submit['day'] == target_day) &
                                    (submit['hour'] == hour)].index

            if not target_indices.empty:
                source_data_filter = (train['building_number'] == building_num) & (train['hour'] == hour)

                month_day_conditions = []
                for src_date_info in source_dates:
                    month_day_conditions.append(
                        (train['month'] == src_date_info['month']) &
                        (train['day'].isin(src_date_info['days']))
                    )

                if building_num == 29:
                    final_source_filter = source_data_filter & (month_day_conditions[0] | month_day_conditions[1])
                else:
                    final_source_filter = source_data_filter & (month_day_conditions[0] | month_day_conditions[1] | month_day_conditions[2])

                values = train[final_source_filter]['power_consumption'].values

                if len(values) > 2:
                    trimmed_mean = (values.sum() - values.max() - values.min()) / (len(values) - 2)
                elif len(values) == 2:
                    trimmed_mean = values.mean()
                elif len(values) == 1:
                    trimmed_mean = values[0]
                else:
                    trimmed_mean = np.nan

                submit.loc[target_indices, 'answer'] = trimmed_mean

    # 필요없는 열 제거
    submit.drop(columns=['date_time', 'hour', 'day', 'month', 'building_number'], inplace=True)

    output_file_path = f'{OUTPUT_DIR}/final.csv'
    submit.to_csv(output_file_path, index=False)
    print(f"✔ Final post-processed file saved to {output_file_path}")
