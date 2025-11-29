# -*- coding: utf-8 -*-
import numpy as np
import jieba
import re
from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class SatisfactionMultiFactorAnalyzer:
    def __init__(self):
        """初始化多因素满意度分析器"""
        
        # 定义各分析维度的关键词词典
        self.factor_keywords = {
            'environment': [
                '环境', '环保', '绿化', '清洁', '卫生', '污染', '空气', '水质', 
                '生态', '垃圾', '整洁', '干净', '美化', '保护', '治理'
            ],
            'tourist_volume': [
                '游客', '人流', '拥挤', '人数', '客流量', '旺季', '淡季', '排队',
                '爆满', '稀少', '接待', '容量', '饱和', '密集', '稀疏'
            ],
            'transportation': [
                '交通', '公交', '地铁', '停车', '拥堵', '便利', '可达', '路线',
                '车站', '出行', '自驾', '步行', '换乘', '通畅', '堵塞'
            ],
            'scenic_quantity': [
                '景区', '景点', '数量', '丰富', '多样', '单一', '不足', '充足',
                '资源', '开发', '建设', '规模', '密集', '分散', '布局'
            ]
        }
        
        # 情感词典
        self.sentiment_words = {
            'positive': ['好', '优秀', '满意', '赞', '棒', '便捷', '舒适', '优美', '完善', '理想'],
            'negative': ['差', '糟糕', '不满意', '差劲', '不便', '混乱', '脏乱', '欠缺', '不足']
        }
        
    def preprocess_text(self, text):
        """文本预处理"""
        text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
        words = jieba.cut(text)
        words = [word for word in words if len(word) > 1]
        return words
    
    def calculate_factor_scores(self, documents):
        """
        计算各因素得分
        
        Args:
            documents: 文档列表
            
        Returns:
            各因素得分矩阵
        """
        n_docs = len(documents)
        factor_scores = {factor: np.zeros(n_docs) for factor in self.factor_keywords.keys()}
        sentiment_scores = np.zeros(n_docs)
        
        # 计算文档频率
        doc_freq = {factor: defaultdict(int) for factor in self.factor_keywords.keys()}
        
        for i, doc in enumerate(documents):
            words = self.preprocess_text(doc)
            
            # 计算情感得分
            pos_count = sum(1 for word in words if word in self.sentiment_words['positive'])
            neg_count = sum(1 for word in words if word in self.sentiment_words['negative'])
            sentiment_scores[i] = (pos_count - neg_count) / len(words) if words else 0
            
            # 计算各因素词频
            for factor, keywords in self.factor_keywords.items():
                factor_count = sum(1 for word in words if word in keywords)
                factor_scores[factor][i] = factor_count / len(words) if words else 0
                
                # 更新文档频率
                if factor_count > 0:
                    for keyword in keywords:
                        if keyword in words:
                            doc_freq[factor][keyword] += 1
        
        # 计算逆文档频率加权的因素得分
        weighted_factor_scores = {}
        for factor in self.factor_keywords.keys():
            total_idf = 0
            for keyword in self.factor_keywords[factor]:
                df = doc_freq[factor][keyword]
                idf = np.log(n_docs / (df + 1)) + 1  # 加1平滑
                total_idf += idf
            
            # 使用平均IDF进行加权
            avg_idf = total_idf / len(self.factor_keywords[factor])
            weighted_factor_scores[factor] = factor_scores[factor] * avg_idf
        
        return weighted_factor_scores, sentiment_scores
    
    def analyze_relationship(self, documents):
        """
        分析满意度与各因素的关系
        
        Args:
            documents: 文档列表
            
        Returns:
            分析结果字典
        """
        factor_scores, satisfaction_scores = self.calculate_factor_scores(documents)
        
        # 计算各因素的平均重要性
        factor_importance = {}
        for factor, scores in factor_scores.items():
            factor_importance[factor] = np.mean(scores)
        
        # 计算各因素占比
        total_importance = sum(factor_importance.values())
        factor_proportions = {factor: imp/total_importance for factor, imp in factor_importance.items()}
        
        # 进行回归分析
        X = np.array([factor_scores[factor] for factor in self.factor_keywords.keys()]).T
        y = satisfaction_scores
        
        if len(documents) > 1:
            model = LinearRegression()
            model.fit(X, y)
            coefficients = dict(zip(self.factor_keywords.keys(), model.coef_))
            r_squared = model.score(X, y)
        else:
            coefficients = {factor: 0 for factor in self.factor_keywords.keys()}
            r_squared = 0
        
        return {
            'factor_scores': factor_scores,
            'satisfaction_scores': satisfaction_scores,
            'factor_importance': factor_importance,
            'factor_proportions': factor_proportions,
            'regression_coefficients': coefficients,
            'r_squared': r_squared
        }
    
    def generate_recommendations(self, analysis_results):
        """
        基于分析结果生成建议
        
        Args:
            analysis_results: 分析结果
            
        Returns:
            建议字典
        """
        proportions = analysis_results['factor_proportions']
        coefficients = analysis_results['regression_coefficients']
        
        # 按重要性排序
        sorted_factors = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = {
            'priority_factors': [],
            'improvement_suggestions': [],
            'optimal_allocation': {}
        }
        
        # 确定优先因素
        for factor, proportion in sorted_factors[:2]:
            recommendations['priority_factors'].append({
                'factor': factor,
                'proportion': proportion,
                'impact': coefficients.get(factor, 0)
            })
        
        # 生成改进建议
        factor_names = {
            'environment': '环境质量',
            'tourist_volume': '游客数量管理',
            'transportation': '交通便利性', 
            'scenic_quantity': '景区资源'
        }
        
        for factor, proportion in sorted_factors:
            if coefficients.get(factor, 0) > 0:
                recommendations['improvement_suggestions'].append(
                    f"加强{factor_names[factor]}建设（当前关注度：{proportion:.1%}）"
                )
        
        # 计算最优资源分配
        total_coefficient = sum(abs(coef) for coef in coefficients.values())
        if total_coefficient > 0:
            for factor in coefficients:
                weight = abs(coefficients[factor]) / total_coefficient
                recommendations['optimal_allocation'][factor_names[factor]] = weight
        
        return recommendations



def main():
    """主函数"""
    print("=" * 70)
    print("人民满意度与多因素关系分析系统")
    print("=" * 70)
    
    # 创建分析器
    analyzer = SatisfactionMultiFactorAnalyzer()
    
    # 获取样本数据
    documents= 
    print(f"分析文档数量: {len(documents)}")
    
    # 执行分析
    results = analyzer.analyze_relationship(documents)
    
    # 显示分析结果
    print("\n" + "=" * 70)
    print("各因素重要性分析")
    print("=" * 70)
    
    factor_names = {
        'environment': '环境质量',
        'tourist_volume': '游客数量', 
        'transportation': '交通便利',
        'scenic_quantity': '景区数量'
    }
    
    print("\n各因素在评价中的占比:")
    for factor, proportion in results['factor_proportions'].items():
        print(f"  {factor_names[factor]}: {proportion:.2%}")
    
    print("\n回归分析系数 (对满意度的影响程度):")
    for factor, coef in results['regression_coefficients'].items():
        print(f"  {factor_names[factor]}: {coef:.4f}")
    
    print(f"\n模型拟合度 (R²): {results['r_squared']:.4f}")
    
    # 生成建议
    recommendations = analyzer.generate_recommendations(results)
    
    print("\n" + "=" * 70)
    print("优化建议")
    print("=" * 70)
    
    print("\n优先关注因素:")
    for item in recommendations['priority_factors']:
        print(f"  {factor_names[item['factor']]}: 关注度 {item['proportion']:.1%}, 影响系数 {item['impact']:.4f}")
    
    print("\n改进建议:")
    for suggestion in recommendations['improvement_suggestions']:
        print(f"  • {suggestion}")
    
    print("\n建议资源分配比例:")
    for factor, allocation in recommendations['optimal_allocation'].items():
        print(f"  {factor}: {allocation:.1%}")
    
    # 可视化结果
    visualize_results(results, factor_names)

def visualize_results(results, factor_names):
    """可视化分析结果"""
    plt.figure(figsize=(15, 10))
    
    # 因素占比饼图
    plt.subplot(2, 2, 1)
    labels = [factor_names[factor] for factor in results['factor_proportions'].keys()]
    sizes = [results['factor_proportions'][factor] for factor in results['factor_proportions'].keys()]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('各因素在评价中的占比')
    
    # 回归系数柱状图
    plt.subplot(2, 2, 2)
    factors = list(results['regression_coefficients'].keys())
    coefficients = [results['regression_coefficients'][factor] for factor in factors]
    factor_labels = [factor_names[factor] for factor in factors]
    colors = ['green' if coef > 0 else 'red' for coef in coefficients]
    plt.bar(factor_labels, coefficients, color=colors)
    plt.title('各因素对满意度的影响系数')
    plt.xticks(rotation=45)
    plt.ylabel('影响系数')
    
    # 散点图：环境vs满意度
    plt.subplot(2, 2, 3)
    plt.scatter(results['factor_scores']['environment'], results['satisfaction_scores'], alpha=0.6)
    plt.xlabel('环境质量得分')
    plt.ylabel('满意度得分')
    plt.title('环境质量与满意度的关系')
    
    # 散点图：交通vs满意度
    plt.subplot(2, 2, 4)
    plt.scatter(results['factor_scores']['transportation'], results['satisfaction_scores'], alpha=0.6)
    plt.xlabel('交通便利得分')
    plt.ylabel('满意度得分')
    plt.title('交通便利与满意度的关系')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    jieba.initialize()
    main()












    
    

