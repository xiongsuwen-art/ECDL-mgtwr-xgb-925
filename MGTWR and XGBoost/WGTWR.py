import pandas as pd
import numpy as np
from mgtwr.sel import SearchMGTWRParameter
from mgtwr.model import MGTWR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def compare_kernels_and_select_best(coords, t, X, y):
    """
    比较三种核函数并选择最佳参数
    """
    print("=== 核函数参数比较 ===")
    
    kernels = ['gaussian', 'exponential', 'bisquare']
    results = {}
    
    for kernel in kernels:
        print(f"\n测试 {kernel} 核函数...")
        try:
            # 参数搜索
            sel_multi = SearchMGTWRParameter(
                coords, t, X, y, 
                kernel=kernel, 
                fixed=True
            )
            
            bws = sel_multi.search(
                multi_bw_min=[0.1], 
                verbose=True, 
                tol_multi=1.0e-4, 
                time_cost=True
            )
            
            # 使用搜索到的参数拟合模型来获取AIC等指标
            temp_model = MGTWR(
                coords, t, X, y, 
                sel_multi, 
                kernel=kernel, 
                fixed=True
            ).fit()
            
            results[kernel] = {
                'selector': sel_multi,
                'bandwidth': bws,
                'aic': temp_model.aic,
                'aic_c': temp_model.aic_c,
                'bic': temp_model.bic,
                'r2': temp_model.R2,
                'adj_r2': temp_model.adj_R2,
                'success': True
            }
            
            print(f"✓ {kernel} 成功:")
            print(f"  带宽: {bws}")
            print(f"  AIC: {results[kernel]['aic']:.4f}")
            print(f"  AIC_c: {results[kernel]['aic_c']:.4f}")
            print(f"  BIC: {results[kernel]['bic']:.4f}")
            print(f"  R²: {results[kernel]['r2']:.4f}")
            
        except Exception as e:
            print(f"✗ {kernel} 失败: {e}")
            results[kernel] = {
                'success': False, 
                'error': str(e),
                'bandwidth': None,
                'aic': None,
                'aic_c': None,
                'bic': None,
                'r2': None,
                'adj_r2': None
            }
    
    # 选择最佳核函数（基于AIC_c优先）
    successful = {k: v for k, v in results.items() if v['success']}
    if not successful:
        return None, None, results
    
    # 优先使用AIC_c进行选择
    def get_criterion(kernel_data):
        if kernel_data['aic_c'] is not None:
            return kernel_data['aic_c']
        elif kernel_data['aic'] is not None:
            return kernel_data['aic']
        else:
            return float('inf')
    
    best_kernel = min(successful.keys(), key=lambda k: get_criterion(successful[k]))
    
    print(f"\n=== 核函数比较结果 ===")
    print("-" * 80)
    print(f"{'核函数':<12} {'AIC':<12} {'AIC_c':<12} {'BIC':<12} {'R²':<12}")
    print("-" * 80)
    
    for kernel in kernels:
        if results[kernel]['success']:
            aic = results[kernel]['aic']
            aic_c = results[kernel]['aic_c'] 
            bic = results[kernel]['bic']
            r2 = results[kernel]['r2']
            marker = " *" if kernel == best_kernel else ""
            print(f"{kernel:<12} {aic:<12.4f} {aic_c:<12.4f} {bic:<12.4f} {r2:<12.4f}{marker}")
        else:
            print(f"{kernel:<12} {'Failed':<12} {'Failed':<12} {'Failed':<12} {'Failed':<12}")
    
    print("-" * 80)
    print(f"最佳核函数: {best_kernel} (标记为 *)")
    
    # 保存详细比较结果
    comparison_data = []
    for kernel, result in results.items():
        comparison_data.append({
            'kernel': kernel,
            'success': result['success'],
            'bandwidth': str(result['bandwidth']) if result['bandwidth'] is not None else 'Failed',
            'aic': result['aic'],
            'aic_c': result['aic_c'],
            'bic': result['bic'],
            'r2': result['r2'],
            'adj_r2': result['adj_r2'],
            'is_best': kernel == best_kernel if result['success'] else False,
            'error': result.get('error', '') if not result['success'] else ''
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('kernel_comparison_detailed.csv', index=False)
    print("✓ 详细核函数比较结果已保存到 kernel_comparison_detailed.csv")
    
    return best_kernel, successful[best_kernel]['selector'], results

def run_complete_mgtwr_analysis():
    """
    完整的MGTWR分析 - 在原代码基础上添加核函数比较
    """
    print("=== MGTWR完整分析 ===")
    
    # 1. 数据加载
    try:
        data = pd.read_csv(r'E:\04.nothirteen\13data\7GTWR\3.run\MGTWR_ECDL10km.csv')
        print(f"✓ 数据加载成功，形状: {data.shape}")
        print(f"列名: {list(data.columns)}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None
    
    # 2. 数据准备
    coords = data[['longitude', 'latitude']].values
    t = data[['t']].values
    X = data[['x1', 'x2', 'x3']].values
    y = data[['y']].values
    
    print(f"\n数据准备:")
    print(f"坐标形状: {coords.shape}")
    print(f"时间形状: {t.shape}")
    print(f"特征形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    # 3. 核函数比较（新增）
    best_kernel, best_selector, kernel_results = compare_kernels_and_select_best(coords, t, X, y)
    
    if best_kernel is None:
        print("✗ 所有核函数都失败，使用原始高斯核")
        # 回退到原始方法
        sel_multi = SearchMGTWRParameter(
            coords, t, X, y, 
            kernel='gaussian', 
            fixed=True
        )
        bws = sel_multi.search(
            multi_bw_min=[0.1], 
            verbose=True, 
            tol_multi=1.0e-4, 
            time_cost=True
        )
        best_kernel = 'gaussian'
        best_selector = sel_multi
        print(f"✓ 回退到高斯核，带宽: {bws}")
    
    # 4. 模型拟合（使用最佳核函数）
    print(f"\n=== 使用{best_kernel}核函数进行模型拟合 ===")
    try:
        mgtwr = MGTWR(
            coords, t, X, y, 
            best_selector, 
            kernel=best_kernel, 
            fixed=True
        ).fit()
        
        print("✓ 模型拟合成功")
        
    except Exception as e:
        print(f"✗ 模型拟合失败: {e}")
        return None
    
    # 5. 获取预测值和模型结果
    print("\n=== 模型结果分析 ===")
    
    y_pred = mgtwr.predict_value.flatten()
    y_true = y.flatten()
    residuals = mgtwr.reside.flatten()
    
    print(f"预测值形状: {mgtwr.predict_value.shape}")
    print(f"残差形状: {mgtwr.reside.shape}")
    
    # 基本统计信息
    print(f"\n=== 模型统计信息 ===")
    print(f"使用核函数: {best_kernel}")
    print(f"R²: {mgtwr.R2:.4f}")
    print(f"调整R²: {mgtwr.adj_R2:.4f}")
    print(f"AIC: {mgtwr.aic:.4f}")
    print(f"AIC_c: {mgtwr.aic_c:.4f}")
    print(f"BIC: {mgtwr.bic:.4f}")
    print(f"对数似然: {mgtwr.llf:.4f}")
    print(f"有效参数数: {mgtwr.ENP:.4f}")
    print(f"残差平方和: {mgtwr.RSS:.4f}")
    
    # 计算额外的评估指标
    print(f"\n=== 评估指标 ===")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 残差分析
    print(f"\n=== 残差分析 ===")
    print(f"残差均值: {np.mean(residuals):.6f}")
    print(f"残差标准差: {np.std(residuals):.4f}")
    print(f"残差范围: [{np.min(residuals):.4f}, {np.max(residuals):.4f}]")
    
    # 6. 参数分析
    print(f"\n=== 参数分析 ===")
    betas = mgtwr.betas
    bse = mgtwr.bse
    t_values = mgtwr.t_values
    
    print(f"回归系数形状: {betas.shape}")
    print(f"标准误差形状: {bse.shape}")
    print(f"t统计量形状: {t_values.shape}")
    
    # 显示参数统计
    for i in range(betas.shape[1]):
        print(f"\n参数 {i+1} (包含截距):")
        print(f"  系数范围: [{np.min(betas[:, i]):.4f}, {np.max(betas[:, i]):.4f}]")
        print(f"  系数均值: {np.mean(betas[:, i]):.4f}")
        print(f"  系数标准差: {np.std(betas[:, i]):.4f}")
        
        bse_col = bse[:, i]
        valid_bse = bse_col[~np.isnan(bse_col)]
        if len(valid_bse) > 0:
            print(f"  标准误差均值: {np.mean(valid_bse):.4f}")
            print(f"  标准误差范围: [{np.min(valid_bse):.4f}, {np.max(valid_bse):.4f}]")
        else:
            print(f"  标准误差: 全部为NaN")
    
    # 7. 带宽分析
    print(f"\n=== 带宽分析 ===")
    print(f"空间带宽: {mgtwr.bws}")
    print(f"时间带宽: {mgtwr.taus}")
    print(f"时空带宽: {mgtwr.bw_ts}")
    
    # 8. 创建结果DataFrame
    print(f"\n=== 创建结果数据集 ===")
    results_df = pd.DataFrame({
        'longitude': coords[:, 0],
        'latitude': coords[:, 1],
        'time': t.flatten(),
        'actual': y_true,
        'predicted': y_pred,
        'residuals': residuals,
        'kernel_used': best_kernel
    })
    
    # 添加局部回归系数
    for i in range(betas.shape[1]):
        results_df[f'beta_{i}'] = betas[:, i]
    
    # 添加标准误差
    for i in range(bse.shape[1]):
        results_df[f'bse_{i}'] = bse[:, i]
    
    # 添加t统计量
    for i in range(t_values.shape[1]):
        results_df[f't_value_{i}'] = t_values[:, i]
    
    # 保存结果
    output_path = 'mgtwr_complete_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"✓ 完整结果已保存到: {output_path}")
    
    # 9. 可视化
    create_comprehensive_plots(y_true, y_pred, residuals, coords, betas, best_kernel)
    
    # 10. 模型诊断
    print(f"\n=== 模型诊断 ===")
    
    spatial_var = np.var(betas, axis=0)
    print(f"参数空间变异性: {spatial_var}")
    
    temporal_var = np.var(betas, axis=0)
    print(f"参数时间变异性: {temporal_var}")
    
    return mgtwr, results_df

def create_comprehensive_plots(y_true, y_pred, residuals, coords, betas, kernel_used='gaussian'):
    """
    创建综合诊断图
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        fig.suptitle(f'MGTWR Analysis Results (Kernel: {kernel_used.upper()})', fontsize=16)
        
        # 1. 真实值vs预测值
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('ACTUAL')
        axes[0, 0].set_ylabel('PREDICT')
        axes[0, 0].set_title('ACTUAL- PREDICT')
        axes[0, 0].grid(True, alpha=0.3)
        
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        axes[0, 0].text(0.05, 0.95, f'CORRELATION: {correlation:.3f}', 
                       transform=axes[0, 0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 2. 残差分布
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(residuals), color='red', linestyle='--', 
                          label=f'AVERAGE: {np.mean(residuals):.4f}')
        axes[0, 1].set_xlabel('residuals')
        axes[0, 1].set_ylabel('frequent')
        axes[0, 1].set_title('Residual distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差vs预测值
        axes[0, 2].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 2].axhline(y=0, color='r', linestyle='--')
        axes[0, 2].set_xlabel('PREDICTION')
        axes[0, 2].set_ylabel('residuals')
        axes[0, 2].set_title('residuals vs PREDICTION')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 参数空间分布 - 截距项
        scatter = axes[1, 0].scatter(coords[:, 0], coords[:, 1], c=betas[:, 0], 
                                   cmap='viridis', s=20, alpha=0.7)
        axes[1, 0].set_xlabel('longitude')
        axes[1, 0].set_ylabel('latitude')
        axes[1, 0].set_title('BETA_0')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # 5. 参数空间分布 - 第一个变量
        if betas.shape[1] > 1:
            scatter = axes[1, 1].scatter(coords[:, 0], coords[:, 1], c=betas[:, 1], 
                                       cmap='viridis', s=20, alpha=0.7)
            axes[1, 1].set_xlabel('longitude')
            axes[1, 1].set_ylabel('latitude')
            axes[1, 1].set_title('BETA_1')
            plt.colorbar(scatter, ax=axes[1, 1])
        
        # 6. 参数空间分布 - 第二个变量
        if betas.shape[1] > 2:
            scatter = axes[1, 2].scatter(coords[:, 0], coords[:, 1], c=betas[:, 2], 
                                       cmap='viridis', s=20, alpha=0.7)
            axes[1, 2].set_xlabel('longitude')
            axes[1, 2].set_ylabel('latitude')
            axes[1, 2].set_title('BETA_2')
            plt.colorbar(scatter, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('mgtwr_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ 综合分析图已保存为 mgtwr_comprehensive_analysis.png")
        
    except Exception as e:
        print(f"✗ 创建综合分析图失败: {e}")

if __name__ == "__main__":
    # 运行完整分析
    result = run_complete_mgtwr_analysis()
    
    if result is not None:
        mgtwr, results_df = result
        
        print("\n=== 分析完成 ===")
        print("生成的文件:")
        print("1. mgtwr_complete_results.csv - 完整结果数据")
        print("2. mgtwr_comprehensive_analysis.png - 综合分析图")
        print("3. kernel_comparison_detailed.csv - 核函数比较结果")
