import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn.cluster import KMeans
from config import DATA_DIR

def Preprocessing(summer = False, cluster = False):
    train          = pd.read_csv(f'{DATA_DIR}/train.csv', encoding='utf-8-sig')
    test           = pd.read_csv(f'{DATA_DIR}/test.csv',  encoding='utf-8-sig')
    building_info  = pd.read_csv(f'{DATA_DIR}/building_info.csv', encoding='utf-8-sig')

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
    train.drop('num_date_time', axis = 1, inplace=True)

    test = test.rename(columns={
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
    test.drop('num_date_time', axis = 1, inplace=True)

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
        '건물기타': 'Other Buildings',
        '공공': 'Public',
        '학교': 'University',
        '백화점': 'Department Store',
        '병원': 'Hospital',
        '상용': 'Commercial',
        '아파트': 'Apartment',
        '연구소': 'Research Institute',
        'IDC(전화국)': 'IDC',
        '호텔': 'Hotel'
    }

    building_info['building_type'] = building_info['building_type'].replace(translation_dict)
    building_info['solar_power_utility'] = np.where(building_info.solar_power_capacity !='-',1,0)
    building_info['ess_utility'] = np.where(building_info.ess_capacity !='-',1,0)

    train = pd.merge(train, building_info, on='building_number', how='left')
    test = pd.merge(test, building_info, on='building_number', how='left')

    train['date_time'] = pd.to_datetime(train['date_time'], format='%Y%m%d %H')

    # Datetime
    train['hour'] = train['date_time'].dt.hour
    train['day'] = train['date_time'].dt.day
    train['month'] = train['date_time'].dt.month
    train['day_of_week'] = train['date_time'].dt.dayofweek
    test['date_time'] = pd.to_datetime(test['date_time'], format='%Y%m%d %H')

    test['hour'] = test['date_time'].dt.hour
    test['day'] = test['date_time'].dt.day
    test['month'] = test['date_time'].dt.month
    test['day_of_week'] = test['date_time'].dt.dayofweek

    # Calculate 'day_temperature'
    def calculate_day_values(dataframe, target_column, output_column, aggregation_func):
        result_dict = {}

        grouped_temp = dataframe.groupby(['building_number', 'month', 'day'])[target_column].agg(aggregation_func)

        for (building, month, day), value in grouped_temp.items():
            result_dict.setdefault(building, {}).setdefault(month, {})[day] = value

        dataframe[output_column] = [
            result_dict.get(row['building_number'], {}).get(row['month'], {}).get(row['day'], None)
            for _, row in dataframe.iterrows()
        ]

    train['day_max_temperature'] = 0.0
    train['day_mean_temperature'] = 0.0

    calculate_day_values(train, 'temperature', 'day_max_temperature', 'max')
    calculate_day_values(train, 'temperature', 'day_mean_temperature', 'mean')
    calculate_day_values(train, 'temperature', 'day_min_temperature', 'min')

    train['day_temperature_range'] = train['day_max_temperature'] - train['day_min_temperature']

    calculate_day_values(test, 'temperature', 'day_max_temperature', 'max')
    calculate_day_values(test, 'temperature', 'day_mean_temperature', 'mean')
    calculate_day_values(test, 'temperature', 'day_min_temperature', 'min')

    test['day_temperature_range'] = test['day_max_temperature'] - test['day_min_temperature']

    # Outlier
    outlier_idx = train.index[train['power_consumption'] == 0].tolist()

    train.drop(index=outlier_idx, inplace=True)

    outlier_df = pd.read_excel(f'{DATA_DIR}/outlier (4).xlsx')
    outlier_df['date'] = pd.to_datetime(outlier_df['date'], format='%Y%m%d')

    initial_train_rows = train.shape[0]

    for _, row in outlier_df.iterrows():
        building_num = row['num']
        outlier_date = row['date'].date()

        indices_to_drop = train[(train['building_number'] == building_num) &
                                (train['date_time'].dt.date == outlier_date)].index

        train.drop(indices_to_drop, inplace=True)

    rows_dropped_from_outlier_file = initial_train_rows - train.shape[0]

    # Holiday
    holi_weekday = ['2024-06-06', '2024-08-15']

    train['holiday'] = np.where((train.day_of_week >= 5) | (train.date_time.dt.strftime('%Y-%m-%d').isin(holi_weekday)), 1, 0)
    test['holiday'] = np.where((test.day_of_week >= 5) | (test.date_time.dt.strftime('%Y-%m-%d').isin(holi_weekday)), 1, 0)

    # Datetime Fourier transform
    train['sin_hour'] = np.sin(2 * np.pi * train['hour']/23.0)
    train['cos_hour'] = np.cos(2 * np.pi * train['hour']/23.0)
    test['sin_hour'] = np.sin(2 * np.pi * test['hour']/23.0)
    test['cos_hour'] = np.cos(2 * np.pi * test['hour']/23.0)

    train['sin_date'] = -np.sin(2 * np.pi * (train['month']+train['day']/31)/12)
    train['cos_date'] = -np.cos(2 * np.pi * (train['month']+train['day']/31)/12)
    test['sin_date'] = -np.sin(2 * np.pi * (test['month']+test['day']/31)/12)
    test['cos_date'] = -np.cos(2 * np.pi * (test['month']+test['day']/31)/12)

    train['sin_month'] = -np.sin(2 * np.pi * train['month']/12.0)
    train['cos_month'] = -np.cos(2 * np.pi * train['month']/12.0)
    test['sin_month'] = -np.sin(2 * np.pi * test['month']/12.0)
    test['cos_month'] = -np.cos(2 * np.pi * test['month']/12.0)

    train['sin_dayofweek'] = -np.sin(2 * np.pi * (train['day_of_week']+1)/7.0)
    train['cos_dayofweek'] = -np.cos(2 * np.pi * (train['day_of_week']+1)/7.0)
    test['sin_dayofweek'] = -np.sin(2 * np.pi * (test['day_of_week']+1)/7.0)
    test['cos_dayofweek'] = -np.cos(2 * np.pi * (test['day_of_week']+1)/7.0)

    # Summer feature
    if summer == True:
        def summer_cos(date):
            start_date = datetime.strptime("2024-06-01 00:00:00", "%Y-%m-%d %H:%M:%S")
            end_date = datetime.strptime("2024-09-14 00:00:00", "%Y-%m-%d %H:%M:%S")

            period = (end_date - start_date).total_seconds()

            return math.cos(2 * math.pi * (date - start_date).total_seconds() / period)

        def summer_sin(date):
            start_date = datetime.strptime("2024-06-01 00:00:00", "%Y-%m-%d %H:%M:%S")
            end_date = datetime.strptime("2024-09-14 00:00:00", "%Y-%m-%d %H:%M:%S")

            period = (end_date - start_date).total_seconds()

            return math.sin(2 * math.pi * (date - start_date).total_seconds() / period)

        train['summer_cos'] = train['date_time'].apply(summer_cos)
        train['summer_sin'] = train['date_time'].apply(summer_sin)

        test['summer_cos'] = test['date_time'].apply(summer_cos)
        test['summer_sin'] = test['date_time'].apply(summer_sin)

    # CDH
    def CDH(xs):
        cumsum = np.cumsum(xs - 26)
        return np.concatenate((cumsum[:11], cumsum[11:] - cumsum[:-11]))

    def calculate_and_add_cdh(dataframe):
        cdhs = []
        for i in range(1, 101):
            temp = dataframe[dataframe['building_number'] == i]['temperature'].values
            cdh = CDH(temp)
            cdhs.append(cdh)
        return np.concatenate(cdhs)

    train['CDH'] = calculate_and_add_cdh(train)
    test['CDH'] = calculate_and_add_cdh(test)
    train['THI'] = 9/5*train['temperature'] - 0.55*(1-train['humidity']/100)*(9/5*train['humidity']-26)+32
    test['THI'] = 9/5*test['temperature'] - 0.55*(1-test['humidity']/100)*(9/5*test['humidity']-26)+32
    train['WCT'] = 13.12 + 0.6125*train['temperature'] - 11.37*(train['windspeed']**
                                                                0.16) + 0.3965*(train['windspeed']**0.16)*train['temperature']
    test['WCT'] = 13.12 + 0.6125*test['temperature'] - 11.37*(test['windspeed']**
                                                                0.16) + 0.3965*(test['windspeed']**0.16)*test['temperature']

    # Calculate 'power_consumption'
    power_mean = pd.pivot_table(train, values='power_consumption', index=['building_number', 'hour', 'day_of_week'], aggfunc=np.mean).reset_index()
    power_mean.columns = ['building_number', 'hour', 'day_of_week', 'day_hour_mean']

    power_std = pd.pivot_table(train, values='power_consumption', index=['building_number', 'hour', 'day_of_week'], aggfunc=np.std).reset_index()
    power_std.columns = ['building_number', 'hour', 'day_of_week', 'day_hour_std']

    power_hour_mean = pd.pivot_table(train, values='power_consumption', index=['building_number', 'hour'], aggfunc=np.mean).reset_index()
    power_hour_mean.columns = ['building_number', 'hour', 'hour_mean']

    power_hour_std = pd.pivot_table(train, values='power_consumption', index=['building_number', 'hour'], aggfunc=np.std).reset_index()
    power_hour_std.columns = ['building_number', 'hour', 'hour_std']

    train = train.merge(power_mean, on=['building_number', 'hour', 'day_of_week'], how='left')
    test = test.merge(power_mean, on=['building_number', 'hour', 'day_of_week'], how='left')

    train = train.merge(power_std, on=['building_number', 'hour', 'day_of_week'], how='left')
    test = test.merge(power_std, on=['building_number', 'hour', 'day_of_week'], how='left')

    train = train.merge(power_hour_mean, on=['building_number', 'hour'], how='left')
    test = test.merge(power_hour_mean, on=['building_number', 'hour'], how='left')

    train = train.merge(power_hour_std, on=['building_number', 'hour'], how='left')
    test = test.merge(power_hour_std, on=['building_number', 'hour'], how='left')

    train = train.reset_index(drop=True)

    # Cluster
    if cluster == True:
        pivot_table = train.pivot_table(
            values='power_consumption',
            index='building_number',
            columns=['day_of_week', 'hour'],
            aggfunc='mean'
        ).fillna(0)

        pivot_table.columns = [f'dow_{dow}_hour_{hour}' for (dow, hour) in pivot_table.columns]

        k = 5
        kmeans = KMeans(n_clusters=k, random_state=2025, n_init=10)
        clusters = kmeans.fit_predict(pivot_table)

        building_info = building_info.set_index('building_number')
        building_info['cluster'] = pd.Series(clusters, index=pivot_table.index)
        building_info = building_info.reset_index()

        train = pd.merge(train, building_info[['building_number', 'cluster']], on='building_number', how='left')
        test = pd.merge(test, building_info[['building_number', 'cluster']], on='building_number', how='left')

        cluster_counts = building_info['cluster'].value_counts().sort_index()
        print("Cluster-wise building count:")
        print(cluster_counts)

        total_buildings = building_info['building_number'].nunique()
        print("\nTotal number of buildings:", total_buildings)

    return train, test