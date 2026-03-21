import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../Data/optimization_take_home.csv')
print(df.columns.tolist())
import pandas as pd
import matplotlib.pyplot as plt


def plot_from_csv(output_csv, input_csv, num_to_plot=10):
    # 1. 加载数据
    df_out = pd.read_csv(output_csv)
    df_in = pd.read_csv(input_csv)

    # 2. 映射正确的列名（对应你提供的：pickup_long, dropoff_long）
    p_lat, p_lng = 'pickup_lat', 'pickup_long'
    d_lat, d_lng = 'dropoff_lat', 'dropoff_long'

    # 3. 关联经纬度信息
    df_merged = df_out.merge(
        df_in[['delivery_id', p_lat, p_lng, d_lat, d_lng]],
        left_on='Delivery ID', right_on='delivery_id'
    )

    # 4. 确定绘图坐标
    df_merged['lat'] = df_merged.apply(lambda x: x[p_lat] if x['Route Point Type'] == 'Pickup' else x[d_lat], axis=1)
    df_merged['lng'] = df_merged.apply(lambda x: x[p_lng] if x['Route Point Type'] == 'Pickup' else x[d_lng], axis=1)

    # 5. 开始绘图
    plt.figure(figsize=(14, 10))
    # 选取前 N 个骑手，或者你可以随机选：routes = df_merged['Route ID'].sample(num_to_plot).unique()
    routes = df_merged['Route ID'].unique()[:num_to_plot]

    # 使用较鲜艳的色系
    cmap = plt.get_cmap('tab20')

    for i, rid in enumerate(routes):
        route_data = df_merged[df_merged['Route ID'] == rid].sort_values('Route Point Index')
        color = cmap(i % 20)

        # 绘制路径线（带方向箭头感）
        plt.plot(route_data['lng'], route_data['lat'], marker='o', markersize=6,
                 label=f'Route {rid} ({len(route_data) // 2} orders)', color=color, alpha=0.8, linewidth=2)

        # 标注该路径的起点 (S) 和 终点 (E)
        start_node = route_data.iloc[0]
        end_node = route_data.iloc[-1]
        plt.text(start_node['lng'], start_node['lat'], f' S{rid}', color=color, fontweight='bold', fontsize=10)
        plt.scatter(end_node['lng'], end_node['lat'], marker='x', color=color, s=100)  # 终点打叉

    plt.xlabel('Longitude (Palo Alto -> San Jose direction)')
    plt.ylabel('Latitude')
    plt.title(f'Route Geometry Analysis: Top {num_to_plot} Dashers')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # 请确保路径正确
    plot_from_csv('../output.csv', '../Data/optimization_take_home.csv')