import pandas as pd
import numpy as np

import openmeteo_requests

import requests_cache
from retry_requests import retry

def process_data(input_csv):
    """
    按天聚合数据信息
    """
    # 读取数据并解析 DateTime 列
    df = pd.read_csv(input_csv, parse_dates=['DateTime'], dayfirst=True)
    df.replace('?', np.nan, inplace=True)
    # 删除包含 NaN 的行，确保所有数据是有效的
    df.dropna(inplace=True)
    # 设置 DateTime 为索引
    df.dropna(inplace=True)
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.date

    numeric_cols = [
    'Global_active_power', 'Global_reactive_power',
    'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 按天分组并应用不同的聚合操作
    agg_dict = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'RR': 'first',   # 假设这些字段一天内是相同的
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }
    daily_df = df.groupby('DateTime').agg(agg_dict).reset_index()
    daily_df['DateTime'] = pd.to_datetime(daily_df['DateTime'])
    return daily_df
    # daily_df.to_csv(output_csv, index=False)


def data_add_weather(daily_df):
    """
    使用 Open-Meteo API 获取温度数据，并与传入的 daily_df 合并。
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 48.78,
        "longitude": 2.29,
        "start_date": "2006-12-17",
        "end_date": "2008-12-31",
        "timezone": "Europe/Paris",
        "daily": "temperature_2m_mean"

    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()

    # 构建 DataFrame
    daily_data = {"DateTime": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s").strftime('%Y-%m-%d'),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s").strftime('%Y-%m-%d'),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_dataframe = pd.DataFrame(data = daily_data)

    merged_df = pd.merge(daily_df, daily_dataframe, on='DateTime', how='left')
    return merged_df

input_csv = '/opt/data/private/hyl/code/ml-work/data/train.csv'
output_csv = '/opt/data/private/hyl/code/ml-work/data/train_new.csv'
processed_df = process_data(input_csv)
merged_df = data_add_weather(processed_df)
merged_df.to_csv(output_csv, index=False)
