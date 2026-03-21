import pandas as pd


def analyze_ready_time_distribution(input_csv):
    df = pd.read_csv(input_csv)
    # 确保转换为 datetime 格式
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['food_ready_time'] = pd.to_datetime(df['food_ready_time'])

    # 修正点：使用 .dt.total_seconds() 访问器
    df['prep_time'] = (df['food_ready_time'] - df['created_at']).dt.total_seconds() / 60.0

    # 定义分段区间
    bins = [0, 20, 30, 999]
    labels = ['0-20min (极快)', '20-30min (正常)','>30min (严重超时)']

    # 进行分段
    df['prep_category'] = pd.cut(df['prep_time'], bins=bins, labels=labels)

    # 统计分布
    stats = df['prep_category'].value_counts().sort_index().reset_index()
    stats.columns = ['备餐时长区间', '订单数量']
    stats['占比'] = stats['订单数量'].apply(lambda x: f"{(x / len(df)) * 100:.1f}%")

    print("--- 207 个订单备餐时间(Prep Time)分布统计 ---")
    print(stats.to_string(index=False))

    avg_prep = df['prep_time'].mean()
    print(f"\n全量订单平均备餐时间: {avg_prep:.2f} 分钟")

    # 额外统计：天生超时的订单（备餐就超过 45 分钟）
    born_overdue = len(df[df['prep_time'] > 45])
    print(f"注定超时订单 (备餐 > 45min): {born_overdue} 个 (占比 {born_overdue / len(df) * 100:.1f}%)")

    return stats


if __name__ == "__main__":
    # 根据你的文件结构，这里使用相对路径
    analyze_ready_time_distribution('../Data/optimization_take_home.csv')