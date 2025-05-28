import torch
import random

def kmeans_with_time_min_max(features, timestamp, cluster_num, alpha=2, max_iteration=30, tol=1e-4):
    """
    对特征进行k-means聚类，并考虑时间戳维度。
    kmeans++初始化和空簇处理仅基于特征维度，不进行归一化。
    
    参数：
    - features: [T, P, D] 的特征张量
    - timestamp: [T] 的时间戳张量/数组（可以是torch.Tensor或numpy转换后的torch.Tensor）
    - cluster_num: 簇的数量
    - alpha: 平衡时间距离和特征距离的权重参数（默认1）
    - max_iteration: 最大迭代次数
    - tol: 收敛阈值，当中心变化小于tol时停止迭代
    
    返回：
    - features_kmeans: [cluster_num, P, D] 聚类中心特征
    - times_kmeans: [cluster_num] 聚类中心时间戳
    - cluster_assignments: [T] 每个样本的聚类分配结果
    """
    features = features.to(dtype=torch.float32)
    # 确保features和timestamp是torch.Tensor
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if not isinstance(timestamp, torch.Tensor):
        timestamp = torch.tensor(timestamp, dtype=torch.float32)
        
    T, P, D = features.shape
    if T <= cluster_num:
        # 如果样本数少于簇数，直接返回
        return features, timestamp[:cluster_num], None
    
    # 展平特征到 [T, P*D]
    features_flattened = features.view(T, P * D)

    # ----------------------------
    # kmeans++ 初始化（只基于特征距离）
    # ----------------------------
    # 1. 随机选择第一个中心
    first_center_idx = random.randint(0, T-1)
    centers_indices = [first_center_idx]
    
    while len(centers_indices) < cluster_num:
        # 计算每个点到已选中心中最近中心的特征距离
        selected_centers = features_flattened[centers_indices]  # [m, PD]
        # [T, m]每个样本到所有已选中心的距离
        dist_to_chosen = torch.cdist(features_flattened, selected_centers, p=2)
        # 选出最近的中心距离
        nearest_dist, _ = dist_to_chosen.min(dim=1)  # [T]
        
        # 根据kmeans++原理使用距离平方分布选取下一个中心
        probs = (nearest_dist ** 2)
        probs_sum = probs.sum()
        if probs_sum.item() == 0:
            # 如果所有点与中心重合，则随机选
            new_center_idx = random.randint(0, T-1)
        else:
            probs = probs / probs_sum
            new_center_idx = torch.multinomial(probs, 1).item()
        
        centers_indices.append(new_center_idx)

    # 初始化的特征与时间中心
    centers_features = features_flattened[centers_indices]  # [cluster_num, PD]
    centers_times = timestamp[centers_indices]              # [cluster_num]

    # ----------------------------
    # kmeans主循环
    # ----------------------------
    for iteration in range(max_iteration):
        # 计算特征距离: [T, cluster_num]
        distances_features = torch.cdist(features_flattened, centers_features, p=2)
        
        # 计算时间距离: [T, cluster_num]
        distances_times = torch.abs(timestamp.unsqueeze(1) - centers_times.unsqueeze(0))
        
        # final_distances: [T, cluster_num]
        final_distances = torch.zeros_like(distances_features)
        
        for t_idx in range(T):
            feat_dists_t = distances_features[t_idx]  # [cluster_num]
            time_dists_t = distances_times[t_idx]      # [cluster_num]
            
            # 对特征距离进行min-max归一化
            f_min, f_max = feat_dists_t.min(), feat_dists_t.max()
            if f_max > f_min:
                norm_feat = (feat_dists_t - f_min) / (f_max - f_min)
            else:
                # 所有距离相等时，归一化结果为0
                norm_feat = torch.zeros_like(feat_dists_t)
            
            # 对时间距离进行min-max归一化
            t_min, t_max = time_dists_t.min(), time_dists_t.max()
            if t_max > t_min:
                norm_time = ((time_dists_t - t_min) / (t_max - t_min)).to(final_distances.device)
            else:
                # 所有时间距离相等时，归一化结果为0
                norm_time = torch.zeros_like(time_dists_t).to(final_distances.device)
            # 计算最终距离
            final_distances[t_idx] = torch.sqrt(norm_feat**2 + alpha*(norm_time**2))
        
        # 根据最终距离分配聚类
        cluster_assignments = final_distances.argmin(dim=1)
        
        # 根据新分配结果更新簇中心
        new_centers_features = torch.zeros_like(centers_features)
        new_centers_times = torch.zeros_like(centers_times)
        
        for i in range(cluster_num):
            cluster_points = features_flattened[cluster_assignments == i]
            cluster_times = timestamp[cluster_assignments == i]
            if cluster_points.size(0) > 0:
                new_centers_features[i] = cluster_points.mean(dim=0)
                new_centers_times[i] = cluster_times.mean()
            else:
                # 若出现空簇，从所有样本中随机选取一个替代（仅基于特征）
                rand_idx = random.randint(0, T-1)
                new_centers_features[i] = features_flattened[rand_idx]
                new_centers_times[i] = timestamp[rand_idx]
        
        # 判断是否收敛
        center_distance_feat = torch.norm(new_centers_features - centers_features, p=2, dim=1).sum()
        center_distance_time = torch.norm(new_centers_times - centers_times, p=2).sum()
        center_distance = center_distance_feat + center_distance_time
        
        centers_features = new_centers_features
        centers_times = new_centers_times
        
        if center_distance <= tol:
            break
    
    # 恢复特征中心形状
    features_kmeans = centers_features.view(cluster_num, P, D)
    times_kmeans = centers_times  # [cluster_num]
    
    return features_kmeans, times_kmeans, cluster_assignments



