# file: evaluator/phase2_evaluator.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

logger = logging.getLogger(__name__)


class Phase2Evaluator:
    """Phase 2 模型评估器"""

    def __init__(self, agent, env, config: Dict):
        self.agent = agent
        self.env = env
        self.config = config
        self.device = next(agent.policy_net.parameters()).device

        # 从配置获取参数
        self.num_nodes = config.get('num_nodes', 28)
        self.request_dim = config.get('request_feat_dim', 24)

        # 评估结果存储
        self.results = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'confusion_matrix': None,
            'action_distribution': None,
            'per_node_accuracy': {}
        }

    def load_model(self, checkpoint_path: str):
        """加载训练好的模型"""
        logger.info(f"📂 加载模型: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型权重
        if 'policy_net' in checkpoint:
            self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        elif 'model_state_dict' in checkpoint:
            self.agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 直接是模型权重
            self.agent.policy_net.load_state_dict(checkpoint)

        self.agent.policy_net.eval()
        logger.info("✅ 模型加载成功，已切换到评估模式")

        return checkpoint.get('epoch', 0), checkpoint.get('val_loss', 0.0)

    def evaluate_on_dataset(self, expert_data_path: str):
        """在数据集上进行评估"""
        logger.info("=" * 60)
        logger.info("🧪 开始数据集评估")
        logger.info("=" * 60)

        # 加载数据集
        from trainer.phase2_il_trainer import ExpertDataset
        dataset = ExpertDataset(expert_data_path)

        if len(dataset) == 0:
            logger.error("❌ 数据集为空")
            return self.results

        # 创建DataLoader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=32,  # 使用小batch_size以便快速评估
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn
        )

        # 开始评估
        all_predictions = []
        all_targets = []
        all_correct = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if batch is None:
                    continue

                states, req_vecs, targets = batch

                # 移到设备
                states = states.to(self.device)
                req_vecs = req_vecs.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                outputs = self.agent.policy_net(
                    x=states.x,
                    edge_index=states.edge_index,
                    edge_attr=states.edge_attr if hasattr(states, 'edge_attr') else None,
                    req_vec=req_vecs,
                    batch=states.batch
                )

                # 获取预测
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                # 处理logits形状
                batch_size = targets.size(0)
                if logits.size(0) == batch_size * self.num_nodes:
                    # 重塑并选择
                    num_actions = logits.size(1)
                    logits = logits.view(batch_size, self.num_nodes, num_actions)

                    # 使用目标动作选择logits
                    if torch.max(targets) < self.num_nodes:
                        expanded_targets = targets.unsqueeze(-1).unsqueeze(-1)
                        expanded_targets = expanded_targets.expand(-1, 1, num_actions)
                        logits_selected = torch.gather(logits, dim=1, index=expanded_targets)
                        logits = logits_selected.squeeze(1)
                    else:
                        # 使用softmax选择
                        logits = logits.mean(dim=1)

                # 预测
                predictions = torch.argmax(logits, dim=1)

                # 计算准确率
                correct = (predictions == targets).float()

                # 保存结果
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_correct.extend(correct.cpu().numpy())

                if (i + 1) % 10 == 0:
                    batch_acc = correct.mean().item()
                    logger.info(f"  Batch {i + 1}/{len(dataloader)}: 准确率 = {batch_acc:.4f}")

        # 计算总体指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_correct = np.array(all_correct)

        # 基础指标
        accuracy = np.mean(all_correct)
        precision, recall, f1 = self._calculate_classification_metrics(all_predictions, all_targets)

        # 保存结果
        self.results['accuracy'] = accuracy
        self.results['precision'] = precision
        self.results['recall'] = recall
        self.results['f1_score'] = f1
        self.results['confusion_matrix'] = confusion_matrix(all_targets, all_predictions)
        self.results['action_distribution'] = self._analyze_action_distribution(all_predictions, all_targets)
        self.results['per_node_accuracy'] = self._calculate_per_node_accuracy(all_predictions, all_targets)

        # 输出结果
        logger.info("=" * 60)
        logger.info("📊 评估结果:")
        logger.info(f"  ✅ 总体准确率: {accuracy:.4f}")
        logger.info(f"  ✅ 精确率 (Precision): {precision:.4f}")
        logger.info(f"  ✅ 召回率 (Recall): {recall:.4f}")
        logger.info(f"  ✅ F1分数: {f1:.4f}")
        logger.info("=" * 60)

        # 详细分类报告
        logger.info("\n📋 详细分类报告:")
        logger.info(classification_report(all_targets, all_predictions, digits=4))

        return self.results

    def evaluate_in_environment(self, num_episodes: int = 50):
        """在真实环境中评估"""
        logger.info("=" * 60)
        logger.info("🎮 开始环境评估")
        logger.info("=" * 60)

        env_results = {
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'avg_steps': 0.0,
            'completion_rate': 0.0,
            'blocking_rate': 0.0,
            'episode_details': []
        }

        total_reward = 0
        total_success = 0
        total_steps = 0
        total_completed = 0
        total_blocked = 0

        for ep in range(num_episodes):
            try:
                # 重置环境
                reset_result = self.env.reset()
                state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

                if self.env.current_request is None:
                    continue

                # 跟踪轨迹
                episode_reward = 0
                episode_steps = 0
                completed = False
                blocked = False
                trajectory = []

                while True:
                    # 获取动作掩码
                    high_mask = self.env.get_high_level_action_mask()
                    low_mask = self.env.get_low_level_action_mask()

                    # 使用模型预测动作
                    with torch.no_grad():
                        # 准备输入
                        state_tensor = self._state_to_tensor(state)
                        req_vec = self._get_request_vector()

                        # 模型预测
                        outputs = self.agent.policy_net(
                            x=state_tensor.x,
                            edge_index=state_tensor.edge_index,
                            edge_attr=state_tensor.edge_attr if hasattr(state_tensor, 'edge_attr') else None,
                            req_vec=req_vec,
                            batch=state_tensor.batch
                        )

                        logits = outputs[0] if isinstance(outputs, tuple) else outputs

                        # 应用掩码
                        logits = logits.cpu().numpy()
                        masked_logits = logits.copy()

                        # 只考虑有效的低层动作
                        valid_actions = np.where(low_mask)[0]
                        if len(valid_actions) > 0:
                            # 将无效动作的概率设为极小值
                            invalid_actions = np.where(~low_mask)[0]
                            masked_logits[invalid_actions] = -1e9

                            # 选择动作
                            if len(masked_logits.shape) > 1:
                                action = np.argmax(masked_logits[0])  # 取第一个batch
                            else:
                                action = np.argmax(masked_logits)
                        else:
                            action = 0  # 默认动作

                    # 执行动作
                    self.env.step_high_level(action)  # 假设高层动作与低层相同
                    step_result = self.env.step_low_level(action)

                    if len(step_result) == 5:
                        next_state, reward, done, truncated, info = step_result
                    else:
                        next_state, reward, done, info = step_result

                    # 更新统计
                    episode_reward += reward
                    episode_steps += 1
                    trajectory.append(action)

                    # 检查是否完成
                    if info.get('request_completed', False):
                        completed = True
                        total_completed += 1

                    if done:
                        if not completed:
                            blocked = True
                            total_blocked += 1
                        break

                    state = next_state

                # 更新总体统计
                total_reward += episode_reward
                total_steps += episode_steps
                if completed:
                    total_success += 1

                # 保存episode详情
                episode_detail = {
                    'episode': ep,
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'completed': completed,
                    'blocked': blocked,
                    'trajectory': trajectory[:10]  # 只保存前10个动作
                }
                env_results['episode_details'].append(episode_detail)

                if (ep + 1) % 10 == 0:
                    logger.info(f"  Episode {ep + 1}/{num_episodes}: "
                                f"Reward={episode_reward:.2f}, "
                                f"Steps={episode_steps}, "
                                f"Completed={completed}")

            except Exception as e:
                logger.error(f"❌ Episode {ep} 失败: {e}")
                continue

        # 计算最终指标
        env_results['success_rate'] = total_success / max(1, num_episodes)
        env_results['avg_reward'] = total_reward / max(1, num_episodes)
        env_results['avg_steps'] = total_steps / max(1, num_episodes)
        env_results['completion_rate'] = total_completed / max(1, num_episodes)
        env_results['blocking_rate'] = total_blocked / max(1, num_episodes)

        # 输出结果
        logger.info("=" * 60)
        logger.info("📊 环境评估结果:")
        logger.info(f"  ✅ 成功率: {env_results['success_rate']:.4f}")
        logger.info(f"  ✅ 平均奖励: {env_results['avg_reward']:.4f}")
        logger.info(f"  ✅ 平均步数: {env_results['avg_steps']:.2f}")
        logger.info(f"  ✅ 完成率: {env_results['completion_rate']:.4f}")
        logger.info(f"  ✅ 阻塞率: {env_results['blocking_rate']:.4f}")
        logger.info("=" * 60)

        return env_results

    def _collate_fn(self, batch):
        """数据批处理函数"""
        from trainer.phase2_il_trainer import Batch as GraphBatch

        states = []
        actions = []
        req_vecs = []

        for item in batch:
            state = item.get('state') or item.get('network_state')
            if state is None:
                continue

            action = item.get('action')
            if action is None:
                continue

            states.append(state)
            actions.append(action)

            if hasattr(state, 'req_vec'):
                req_vecs.append(state.req_vec)
            elif hasattr(state, 'req'):
                req_vecs.append(state.req)
            else:
                req_vecs.append(torch.zeros(self.request_dim))

        if not states:
            return None

        try:
            graph_batch = GraphBatch.from_data_list(states)
            req_vecs = torch.stack(req_vecs, dim=0).float()
            actions = torch.tensor(actions, dtype=torch.long)

            return graph_batch, req_vecs, actions
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            return None

    def _state_to_tensor(self, state):
        """将状态转换为模型输入格式"""
        # 这里需要根据您的环境状态格式进行调整
        # 假设state已经是torch_geometric Data对象
        return state

    def _get_request_vector(self):
        """从环境中获取请求向量"""
        # 这里需要根据您的环境实现调整
        # 返回形状为[1, request_dim]的张量
        return torch.randn(1, self.request_dim).to(self.device)

    def _calculate_classification_metrics(self, predictions, targets):
        """计算分类指标"""
        from sklearn.metrics import precision_score, recall_score, f1_score

        # 计算微观平均（对所有样本平等对待）
        precision = precision_score(targets, predictions, average='micro', zero_division=0)
        recall = recall_score(targets, predictions, average='micro', zero_division=0)
        f1 = f1_score(targets, predictions, average='micro', zero_division=0)

        return precision, recall, f1

    def _analyze_action_distribution(self, predictions, targets):
        """分析动作分布"""
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        unique_targets, target_counts = np.unique(targets, return_counts=True)

        return {
            'prediction_distribution': dict(zip(unique_preds, pred_counts)),
            'target_distribution': dict(zip(unique_targets, target_counts)),
            'prediction_entropy': self._calculate_entropy(pred_counts),
            'target_entropy': self._calculate_entropy(target_counts)
        }

    def _calculate_per_node_accuracy(self, predictions, targets):
        """计算每个节点的准确率"""
        per_node_acc = {}
        for node in range(self.num_nodes):
            mask = targets == node
            if np.sum(mask) > 0:
                node_correct = np.sum(predictions[mask] == targets[mask])
                node_total = np.sum(mask)
                per_node_acc[node] = node_correct / node_total

        return per_node_acc

    def _calculate_entropy(self, counts):
        """计算分布的熵"""
        probs = counts / np.sum(counts)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy

    def visualize_results(self, output_dir: str):
        """可视化评估结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. 混淆矩阵热图
        if self.results['confusion_matrix'] is not None:
            plt.figure(figsize=(10, 8))
            cm = self.results['confusion_matrix']

            # 归一化混淆矩阵
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=range(self.num_nodes),
                        yticklabels=range(self.num_nodes))

            plt.title('Normalized Confusion Matrix')
            plt.xlabel('Predicted Node')
            plt.ylabel('True Node')
            plt.tight_layout()
            plt.savefig(output_path / 'confusion_matrix.png')
            plt.close()

        # 2. 准确率柱状图
        if self.results['per_node_accuracy']:
            plt.figure(figsize=(12, 6))
            nodes = list(self.results['per_node_accuracy'].keys())
            accuracies = list(self.results['per_node_accuracy'].values())

            plt.bar(nodes, accuracies)
            plt.axhline(y=self.results['accuracy'], color='r', linestyle='--',
                        label=f'Overall Accuracy: {self.results["accuracy"]:.4f}')

            plt.title('Per-Node Accuracy')
            plt.xlabel('Node ID')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'per_node_accuracy.png')
            plt.close()

        # 3. 动作分布图
        if self.results['action_distribution']:
            plt.figure(figsize=(14, 6))

            # 预测分布
            plt.subplot(1, 2, 1)
            pred_dist = self.results['action_distribution']['prediction_distribution']
            plt.bar(list(pred_dist.keys()), list(pred_dist.values()))
            plt.title(
                f'Prediction Distribution (Entropy: {self.results["action_distribution"]["prediction_entropy"]:.4f})')
            plt.xlabel('Node')
            plt.ylabel('Count')

            # 目标分布
            plt.subplot(1, 2, 2)
            target_dist = self.results['action_distribution']['target_distribution']
            plt.bar(list(target_dist.keys()), list(target_dist.values()))
            plt.title(f'Target Distribution (Entropy: {self.results["action_distribution"]["target_entropy"]:.4f})')
            plt.xlabel('Node')
            plt.ylabel('Count')

            plt.tight_layout()
            plt.savefig(output_path / 'action_distribution.png')
            plt.close()

        logger.info(f"✅ 可视化结果已保存到: {output_path}")