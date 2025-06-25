import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import random
from deap import base, creator, tools, algorithms
import matplotlib
matplotlib.use('TkAgg')  # 设置为TkAgg
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
warnings.filterwarnings('ignore')


# 数据加载与预处理（限制数据量）
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, nrows=1000)  # 仅加载1000行数据
    df = df.dropna(subset=['PM2.5'])

    df['date_str'] = df['date'].astype(int).astype(str)
    df['year'] = df['date_str'].str[:4].astype(int)
    df['month'] = df['date_str'].str[4:6].astype(int)
    df['day'] = df['date_str'].str[6:8].astype(int)
    df['time_str'] = df['time'].astype(int).astype(str).str.zfill(2)
    df['TIME'] = pd.to_datetime(df['year'].astype(str) + '-' +
                                df['month'].astype(str) + '-' +
                                df['day'].astype(str) + ' ' +
                                df['time_str'] + ':00:00')
    df = df.set_index('TIME')

    df['PM2.5_lag1'] = df['PM2.5'].shift(1)
    df['PM2.5_lag2'] = df['PM2.5'].shift(2)
    df = df.dropna()  # 减少滞后特征数量

    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    return train_data, test_data, df


# ARIMA模型（一次性预测替代逐点预测）
def arima_model(train_data, test_data, order):
    model = ARIMA(train_data['PM2.5'].values, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test_data))
    return predictions


# SVM模型（简化参数范围）
def svm_model(train_data, test_data, params):
    X_train = train_data[['PM2.5_lag1']].values  # 仅使用1阶滞后特征
    y_train = train_data['PM2.5'].values
    X_test = test_data[['PM2.5_lag1']].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVR(C=params[0], gamma=params[1], kernel='rbf', max_iter=500)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# MLP模型（减少网络复杂度）
def mlp_model(train_data, test_data, params):
    X_train = train_data[['PM2.5_lag1']].values
    y_train = train_data['PM2.5'].values
    X_test = test_data[['PM2.5_lag1']].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 确保学习率为正
    learning_rate = max(0.00001, params[1])  # 设置最小值防止接近0

    model = MLPRegressor(
        hidden_layer_sizes=(int(params[0]),),  # 单层隐藏层
        activation='relu',
        solver='adam',
        max_iter=200,  # 减少迭代次数
        learning_rate_init=learning_rate,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


# LSTM模型（大幅减少训练开销）
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def lstm_model(train_data, test_data, params):
    train_pm25 = train_data['PM2.5'].values.reshape(-1, 1)
    test_pm25 = test_data['PM2.5'].values.reshape(-1, 1)

    scaler = StandardScaler()
    train_pm25 = scaler.fit_transform(train_pm25)
    test_pm25 = scaler.transform(test_pm25)

    seq_length = min(int(params[0]), 10)  # 限制序列长度
    batch_size = int(params[1])
    hidden_size = int(params[2])

    # 确保学习率为正
    learning_rate = max(0.00001, params[3])  # 设置最小值防止接近0

    X_train, y_train = create_sequences(train_pm25, seq_length)
    X_test, y_test = create_sequences(test_pm25, seq_length)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)

    model = LSTM(input_size=1, hidden_size=hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cpu')
    model.to(device)

    num_epochs = 10  # 仅训练10轮
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device)).cpu().numpy()

    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()


# 评估指标
def evaluate_model(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


# 遗传算法优化（修复参数变异问题）
def genetic_algorithm_optimization(model_type, train_data, test_data, param_ranges, pop_size=10, generations=5):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    if model_type == 'ARIMA':
        # 为ARIMA注册整数参数生成器
        toolbox.register("attr_p", random.randint, 1, 3)
        toolbox.register("attr_d", random.randint, 0, 1)
        toolbox.register("attr_q", random.randint, 1, 3)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_p, toolbox.attr_d, toolbox.attr_q), n=1)
    elif model_type == 'SVM':
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (lambda: random.uniform(1, 10),
                          lambda: random.uniform(0.01, 0.1)), n=1)
    elif model_type == 'MLP':
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (lambda: random.randint(10, 30),
                          lambda: random.uniform(0.0001, 0.001)), n=1)
    elif model_type == 'LSTM':
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (lambda: random.randint(5, 10),
                          lambda: random.randint(16, 32),
                          lambda: random.randint(10, 30),
                          lambda: random.uniform(0.0001, 0.001)), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalModel(individual):
        if model_type == 'ARIMA':
            # 确保ARIMA参数是整数
            order = (int(individual[0]), int(individual[1]), int(individual[2]))
            predictions = arima_model(train_data, test_data, order)
        elif model_type == 'SVM':
            # 确保SVM参数在有效范围内
            params = [max(0.01, min(100, individual[0])),  # C参数
                      max(0.0001, min(1, individual[1]))]  # gamma参数
            predictions = svm_model(train_data, test_data, params)
        elif model_type == 'MLP':
            # 确保MLP参数在有效范围内
            params = [max(10, min(100, int(individual[0]))),  # 神经元数量
                      max(0.00001, min(0.01, individual[1]))]  # 学习率
            predictions = mlp_model(train_data, test_data, params)
        elif model_type == 'LSTM':
            # 确保LSTM参数在有效范围内
            params = [max(5, min(20, int(individual[0]))),  # 序列长度
                      max(8, min(64, int(individual[1]))),  # batch_size
                      max(10, min(100, int(individual[2]))),  # 隐藏层大小
                      max(0.00001, min(0.01, individual[3]))]  # 学习率
            predictions = lstm_model(train_data, test_data, params)

        true_values = test_data['PM2.5'].values
        if model_type == 'LSTM':
            true_values = true_values[int(params[0]):]  # 使用修正后的参数
        return evaluate_model(true_values, predictions)['RMSE'],

    toolbox.register("evaluate", evalModel)
    toolbox.register("mate", tools.cxTwoPoint)

    # 为ARIMA使用整数变异，其他模型使用浮点变异
    if model_type == 'ARIMA':
        toolbox.register("mutate", tools.mutUniformInt, low=[1, 0, 1], up=[3, 1, 3], indpb=0.2)
    else:
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)

    toolbox.register("select", tools.selTournament, tournsize=2)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations,
                        stats=stats, halloffame=hof, verbose=False)
    return hof[0], None


# 主函数（简化流程）
def main():
    file_path = "./数据/2016wulumuqiconcated_data.csv"
    train_data, test_data, _ = load_and_preprocess_data(file_path)

    models = ['ARIMA', 'SVM', 'MLP', 'LSTM']
    default_params = {
        'ARIMA': (2, 1, 2),
        'SVM': (1.0, 0.1),
        'MLP': (20, 0.001),
        'LSTM': (8, 16, 20, 0.001)
    }

    print("优化前结果:")
    for model in models:
        print(f"\n{model}模型:")
        if model == 'ARIMA':
            predictions = arima_model(train_data, test_data, default_params[model])
        elif model == 'SVM':
            predictions = svm_model(train_data, test_data, default_params[model])
        elif model == 'MLP':
            predictions = mlp_model(train_data, test_data, default_params[model])
        elif model == 'LSTM':
            predictions = lstm_model(train_data, test_data, default_params[model])

        true_values = test_data['PM2.5'].values
        if model == 'LSTM':
            true_values = true_values[int(default_params[model][0]):]
        metrics = evaluate_model(true_values, predictions)

        print(f"参数: {default_params[model]}")
        print(f"RMSE: {metrics['RMSE']:.4f}")

    print("\n\n开始优化参数...")
    optimized_params = {}
    for model in models:
        print(f"\n优化{model}模型:")
        param_ranges = {
            'ARIMA': [(1, 3), (0, 1), (1, 3)],
            'SVM': [(1, 10), (0.01, 0.1)],
            'MLP': [(10, 30), (0.0001, 0.001)],
            'LSTM': [(5, 10), (16, 32), (10, 30), (0.0001, 0.001)]
        }
        best_params, _ = genetic_algorithm_optimization(
            model, train_data, test_data, param_ranges[model], pop_size=10, generations=5
        )
        optimized_params[model] = best_params
        print(f"最优参数: {best_params}")

    # 优化后的结果
    print("\n优化后结果:")
    results = {}
    for model in models:
        print(f"\n{model}模型:")
        if model == 'ARIMA':
            predictions = arima_model(train_data, test_data, optimized_params[model])
        elif model == 'SVM':
            predictions = svm_model(train_data, test_data, optimized_params[model])
        elif model == 'MLP':
            predictions = mlp_model(train_data, test_data, optimized_params[model])
        elif model == 'LSTM':
            predictions = lstm_model(train_data, test_data, optimized_params[model])

        true_values = test_data['PM2.5'].values
        if model == 'LSTM':
            true_values = true_values[int(optimized_params[model][0]):]

        metrics = evaluate_model(true_values, predictions)
        results[f'{model}_after'] = {
            'params': optimized_params[model],
            'metrics': metrics,
            'predictions': predictions
        }

        print(f"参数: {optimized_params[model]}")
        print(f"RMSE: {metrics['RMSE']:.4f}")

    # 可视化结果
    plt.figure(figsize=(15, 10))
    for i, model in enumerate(models, 1):
        plt.subplot(2, 2, i)
        true_values = test_data['PM2.5'].values
        if model == 'LSTM':
            true_values = true_values[int(optimized_params[model][0]):]

        plt.plot(true_values[:50], label='TrueValue', c='blue')
        plt.plot(results[f'{model}_after']['predictions'][:50], label='OptimizeValue', c='red')

        plt.title(f'{model} Prediction')
        plt.xlabel('Time')
        plt.ylabel('PM2.5 concentration')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('pm25_prediction_comparison.png')
    plt.show()


if __name__ == "__main__":
    main()