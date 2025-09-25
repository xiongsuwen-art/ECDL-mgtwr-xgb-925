# 导入工具包
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
import seaborn as sns
import pickle
import os
import glob
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import shap
import matplotlib
import matplotlib.patches
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import xgboost as xgb

# 设置英文字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'

# 坐标轴刻度数字加粗
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# 图例加粗
plt.legend(prop={'weight': 'bold'})

# 自定义颜色
colors =[
    "#2B79B5",  # 紫红色
    "#A9D8E8",  # 酒红色
    "#FFFFBF",  # 橙红色
    "#FCAE60",  # 橙色
    "#D61A1D"   # 亮黄色（最亮处）
]
cmap = LinearSegmentedColormap.from_list("custom", colors)

#定义批处理函数与流程
def process_csv_file(csv_path, output_base_dir):
    """处理单个CSV文件的函数"""
    # 创建输出目录
    file_name = os.path.basename(csv_path).split('.')[0]
    output_dir = os.path.join(output_base_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n开始处理文件: {csv_path}")
    print(f"输出目录: {output_dir}")
    
    # 读取数据
    try:
        features = pd.read_csv(csv_path, encoding='GBK')
    except:
        try:
            features = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            print(f"无法读取文件 {csv_path}: {str(e)}")
            return
    
    print(f"文件 {file_name} 成功读取，包含 {features.shape[0]} 行和 {features.shape[1]} 列")
    
    # 查找目标列（假设目标列包含'MODIS'或'LST'）
    target_cols = [col for col in features.columns if 'CCD' in col or 'Lu_LST' in col]
    if not target_cols:
        print(f"警告: 在文件 {file_name} 中未找到目标列（包含MODIS或LST的列）")
        return
    
    target_col = target_cols[0]  # 使用第一个找到的目标列
    print(f"使用 {target_col} 作为目标变量")
    
    # 独热编码
    features = pd.get_dummies(features)
    
    # 尝试找到需要排除的ID列
    id_cols = [col for col in features.columns if 'Id' in col.upper()]
    drop_cols = id_cols.copy()
    if target_col in features.columns:
        drop_cols.append(target_col)
    
    # 检查是否有可能是LST相关的其他列需要排除
    lst_cols = [col for col in features.columns if 'CCD' in col and col != target_col]
    drop_cols.extend(lst_cols)
    
    # 提取特征和标签
    if target_col in features.columns:
        labels = features[target_col]
        features = features.drop(drop_cols, axis=1)
    else:
        print(f"错误: 目标列 {target_col} 在处理后的数据中不存在")
        return
    
    # 特征名字留着备用
    feature_list = list(features.columns)
    
    # 转换成所需格式
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    # 数据集切分
    train_features, test_features, train_labels, test_labels = train_test_split(
        features_array, labels_array, test_size=0.30, random_state=42)
    
    ## XGBoost调参(以下部分)
    print("开始XGBoost调参...")
    # 创建参数网格
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'n_estimators': [50, 100]
    }

    # 创建XGBoost模型用于GridSearchCV
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        device='cuda',  # 使用GPU
        eval_metric='rmse',
        random_state=42
    )

    # 创建网格搜索
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    # 执行网格搜索
    grid_search.fit(train_features, train_labels)

    # 获取最佳参数
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")

    # 将最佳参数应用到训练
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': best_params['learning_rate'],
        'max_depth': best_params['max_depth'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'device': 'cuda',
        'random_state': 42
    }
    num_round = best_params['n_estimators']

    # 将调参结果保存到文件（调参完成，不想调参从XGBOOST调参开始删下来，这段也要删）
    with open(os.path.join(output_dir, 'best_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'最佳参数:\n{best_params}\n')

    # 将数据转换为XGBoost的DMatrix格式
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dtest = xgb.DMatrix(test_features, label=test_labels)

    # 训练模型
    evals = [(dtrain, 'train'), (dtest, 'test')]
    num_round = 100
    rf = xgb.train(params, dtrain, num_round, evals=evals, verbose_eval=False)
    
    # 预测和评估
    predictions = rf.predict(dtest)
    baseline_errors = abs(predictions - test_labels)
    r2 = r2_score(test_labels, predictions)
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    
    # 保存基本评估结果
    with open(os.path.join(output_dir, 'model_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'平均误差: {round(np.mean(baseline_errors), 2)}\n')
        f.write(f'R²: {r2:.4f}\n')
        f.write(f'MSE: {mse:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
    
    print(f'平均误差: {round(np.mean(baseline_errors), 2)}')
    print(f'R²: {r2:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []   # 保存各折RMSE
    r2_scores = []     # 保存各折R²

    fold = 1
    for train_index, test_index in kf.split(features_array):
        train_feat, test_feat = features_array[train_index], features_array[test_index]
        train_lab, test_lab = labels_array[train_index], labels_array[test_index]
        
        # 转换为DMatrix
        dtrain_fold = xgb.DMatrix(train_feat, label=train_lab)
        dtest_fold = xgb.DMatrix(test_feat, label=test_lab)
        
        # 训练和预测
        bst = xgb.train(params, dtrain_fold, num_round)
        pred = bst.predict(dtest_fold)
        
        mse = mean_squared_error(test_lab, pred)
        rmse = np.sqrt(mse)              # 每折RMSE
        r2 = r2_score(test_lab, pred)    # 每折R²
        
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
        fold += 1

    # 计算平均性能指标
    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    avg_r2 = np.mean(r2_scores)

    with open(os.path.join(output_dir, 'cross_validation_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"各折MSE: {[f'{mse:.4f}' for mse in mse_scores]}\n")
        f.write(f"平均MSE: {avg_mse:.4f} ± {std_mse:.4f}\n")
        f.write(f"各折RMSE: {[f'{rmse:.4f}' for rmse in rmse_scores]}\n")
        f.write(f"平均RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}\n")
        f.write(f"各折R2: {[f'{r2:.4f}' for r2 in r2_scores]}\n")
        f.write(f"平均R2: {avg_r2:.4f}\n")

    # 在训练后的模型后添加特征重要性分析 (添加在预测评估部分之后)
    importance = rf.get_score(importance_type='gain')  # 使用信息增益作为重要性指标
    importance_sum = sum(importance.values())
    importance_norm = {k: v / importance_sum for k, v in importance.items()}
    sorted_importance = sorted(importance_norm.items(), key=lambda x: x[1], reverse=True)
    importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    print(f"特征重要性已保存到 {output_dir}/feature_importance.csv")

    # SHAP分析部分 - 使用GPU加速
    try:
        X_sample_df = pd.DataFrame(test_features, columns=feature_list)
        X_array = X_sample_df.values
        feature_names = X_sample_df.columns.tolist()    
        explainer = shap.TreeExplainer(rf)
        print("开始计算整体SHAP值")  
        shap_values = explainer.shap_values(X_array)
        print("整体SHAP值计算完成")

        # 1. SHAP摘要图（条形图），以下出图部分plt.title部分的红字都是可以改的，同一行的蓝字file_name是输入csv的文件名，plt.savefig那行的'shap_bar_{file_name}.png'是保存的文件名，
        # figure_size是指尺寸，这部分建议结合实际情况对照着改
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_array, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"{file_name}-abstract", fontsize=16)
        plt.xlabel("SHAP value", fontsize=14)
        plt.ylabel("feature", fontsize=14)
        ax = plt.gca()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.tick_params(axis='both', direction='in')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_bar_{file_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SHAP摘要图（蜂群图）
        plt.figure(figsize=(14, 10))
        # 显示全部特征
        shap.summary_plot(
        shap_values, 
        X_array, 
        feature_names=feature_names, 
        show=False, 
        cmap=cmap, 
        max_display=len(feature_names)  # 显示所有特征
        )
        plt.title(f"{file_name}-SHAP Beeswarm Plot", fontsize=16)
        ax = plt.gca()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)       
        ax.tick_params(axis='both', direction='in')

        # 添加0刻度虚线
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7) 
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'scatter_{file_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. SHAP依赖图：前6个重要特征，6可以改，改[:6]这个地方
        important_indices = np.argsort(-np.abs(shap_values).mean(0))[:6]
        for i in important_indices:
            feature_name = feature_list[i]
            plt.figure(figsize=(10, 7))
            feature_values = X_sample_df.iloc[:, i].values
            shap_values_for_feature = shap_values[:, i]

            correlations = []
            for j in range(X_sample_df.shape[1]):
                if i != j:  # 排除自己
                    corr = abs(np.corrcoef(shap_values_for_feature, X_sample_df.iloc[:, j])[0, 1])
                    correlations.append((j, corr))

            other_ind = sorted(correlations, key=lambda x: x[1], reverse=True)[0][0]
            interaction_values = X_sample_df.iloc[:, other_ind].values
            interaction_name = X_sample_df.columns[other_ind]

            norm = Normalize(vmin=np.min(interaction_values), vmax=np.max(interaction_values))

            sc = plt.scatter(feature_values, shap_values_for_feature, 
                    c=interaction_values, 
                    cmap=cmap, 
                    norm=norm,
                    s=10, 
                    alpha=0.8)
                
            cbar = plt.colorbar(sc)
            cbar.set_label(interaction_name)

            plt.xlabel(feature_name)
            plt.ylabel(f'SHAP value for {feature_name}')
            plt.title(f"{file_name} - {feature_name} rely", fontsize=16)

            ax = plt.gca()
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.tick_params(axis='both', direction='in')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{feature_name}_rely_{file_name}.png"), dpi=300)
            plt.close()
        
        # 4. SHAP瀑布图 - 为多个样本生成
        # 定义颜色
        default_pos_color = "#ff0051"  # SHAP默认正值颜色
        default_neg_color = "#008bfb"  # SHAP默认负值颜色  
        positive_color = "#D61A1D"     # 自定义正值颜色 - 淡黄色
        negative_color = "#2B79B5"     # 自定义负值颜色 - 深紫色

        # 生成前3个样本的瀑布图
        num_waterfall_samples = min(3, X_array.shape[0])
        # 瀑布图部分修改
        for sample_idx in range(num_waterfall_samples):
            plt.figure(figsize=(12, 10))
    
            # 使用X_sample_df获取样本（保持使用DataFrame）
            sample_data = X_sample_df.iloc[sample_idx:sample_idx+1]
            # 转换为numpy数组后传给explainer
            sample_shap_values = explainer.shap_values(sample_data.values)[0]
            expected_value = explainer.expected_value
    
            # 创建瀑布图（保持使用DataFrame的values）
            shap.plots.waterfall(shap.Explanation(
                values=sample_shap_values, 
                base_values=expected_value, 
                data=sample_data.values[0],  # 正确：从DataFrame获取values
                feature_names=feature_names
            ), show=False, max_display=min(8, len(feature_names)))

            # 修改颜色
            for fc in plt.gcf().get_children():
                for fcc in fc.get_children():
                    if (isinstance(fcc, matplotlib.patches.FancyArrow)):
                        if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                            fcc.set_facecolor(positive_color)
                        elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                            fcc.set_color(negative_color)
                    elif (isinstance(fcc, plt.Text)):
                        if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                            fcc.set_color(positive_color)
                        elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                            fcc.set_color(negative_color)

            plt.title(f"{file_name} Waterfall Plot - {sample_idx+1}", fontsize=16)
            ax = plt.gca()
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.tick_params(axis='both', direction='in')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"样本{sample_idx+1}waterfall_{file_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()


        # 导出SHAP值数据
        shap_df = pd.DataFrame(shap_values, columns=feature_list)
        shap_df.to_excel(os.path.join(output_dir, f'shap_value{file_name}.xlsx'), index=False)
        
        # 导出特征重要性排序后的SHAP值
        mean_shap_values = np.abs(shap_values).mean(0)
        shap_importance = pd.DataFrame({
            'Feature': feature_list,
            'SHAP Value': mean_shap_values
        }).sort_values('SHAP Value', ascending=False)
        shap_importance.to_excel(os.path.join(output_dir, f'shap_sort{file_name}.xlsx'), index=False)
        
        # 导出依赖图数据
        dmatrix = xgb.DMatrix(X_array, feature_names=feature_names)
        predictions = rf.predict(dmatrix)
        for i in important_indices:
            feature_name = feature_list[i]
            dependency_df = pd.DataFrame({
                f'{feature_name}_Value': X_sample_df.iloc[:, i].values,
                f'{feature_name}_SHAP Value': shap_values[:, i],
                'Predictions': predictions
            })
            dependency_df.to_excel(os.path.join(output_dir, f'{feature_name} rely {file_name}.xlsx'), index=False)

    except Exception as e:
        print(f"SHAP分析出错: {str(e)}")
    
    print(f"✅ 文件 {file_name} 处理完成！所有结果已保存到 {output_dir} 目录")
####################################

def main():
    # 指定输入目录和输出基础目录
    input_dir = r"E:\test"
    output_base_dir = r"E:\test"
    
    # 确保输出基础目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 获取目录下所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"错误: 在 {input_dir} 目录中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个CSV文件
    for i, csv_path in enumerate(csv_files):
        print(f"\n处理文件 {i+1}/{len(csv_files)}: {os.path.basename(csv_path)}")
        process_csv_file(csv_path, output_base_dir)
    
    print("\n批量处理完成！")


if __name__ == "__main__":
    main()
