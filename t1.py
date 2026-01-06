#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase3 诊断脚本（完整版）
"""

import sys

sys.path.append('.')

import numpy as np
from collections import defaultdict, Counter
import yaml
import pickle
import os

from envs.sfc_env3 import SFC_HIRL_Env
from core.hrl.agent import HRLAgent


def diagnose_phase3_training(env, agent, num_episodes=10):
    """诊断Phase3训练问题"""
    print("=" * 80)
    print("🔬 Phase3 训练诊断工具")
    print("=" * 80)

    stats = {
        'total_episodes': 0,
        'total_steps': 0,
        'successful_episodes': 0,
        'failed_episodes': 0,
        'action_stay': 0,
        'action_move': 0,
        'action_by_node': defaultdict(int),
        'vnf_deploy_attempts': 0,
        'vnf_deploy_success': 0,
        'vnf_deploy_failed': 0,
        'episodes_no_vnf': 0,
        'episodes_partial_vnf': 0,
        'episodes_all_vnf': 0,
        'errors': Counter(),
    }

    print(f"\n📊 开始诊断 {num_episodes} 个episodes...\n")

    for ep in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        obs, info = env.reset()

        ep_stats = {
            'steps': 0,
            'stay_count': 0,
            'move_count': 0,
            'vnf_deployed': 0,
            'vnf_total': 0,
            'success': False,
        }

        req = env.current_request
        if req is None:
            print("⚠️ 没有请求，跳过")
            continue

        print(f"📋 请求 ID={req.get('id')}, VNF={req.get('vnf')}, 目的地={req.get('dest')}")
        ep_stats['vnf_total'] = len(req.get('vnf', []))

        done = False
        step_count = 0
        max_steps = 200

        while not done and step_count < max_steps:
            step_count += 1
            current_node = env.current_node_location
            deployed_vnf = len(env.current_tree.get('placement', {}))

            try:
                # 🔥 获取动作掩码
                action_mask = step_info.get('action_mask', None) if 'step_info' in locals() else info.get('action_mask',
                                                                                                          None)
                if action_mask is None and hasattr(env, 'get_low_level_action_mask'):
                    action_mask = env.get_low_level_action_mask()

                # 选择动作（传入掩码）
                if action_mask is not None:
                    action = agent.select_action(obs, action_mask=action_mask)
                else:
                    action = agent.select_action(obs)

                # 🔥 处理多种HRL动作格式
                if isinstance(action, dict):
                    # 格式1: {'high': subgoal, 'low': node_id}
                    target_node = action.get('low', current_node)
                elif isinstance(action, (tuple, list)):
                    # 格式2: (high_action, low_action)
                    if len(action) >= 2:
                        high_action, low_action = action[0], action[1]
                        target_node = int(low_action)
                    else:
                        print(f"❌ 动作tuple长度异常: {len(action)}")
                        break
                elif isinstance(action, (int, np.integer)):
                    # 格式3: 直接是节点ID
                    target_node = int(action)
                else:
                    print(f"❌ 未知动作格式: {type(action)}, 值={action}")
                    break

            except Exception as e:
                print(f"❌ 动作选择失败: {e}")
                import traceback
                traceback.print_exc()
                break

            # 统计
            if target_node == current_node:
                ep_stats['stay_count'] += 1
                stats['action_stay'] += 1
                action_type = "STAY"
            else:
                ep_stats['move_count'] += 1
                stats['action_move'] += 1
                action_type = "MOVE"

            stats['action_by_node'][target_node] += 1

            if step_count % 5 == 0 or step_count <= 3:
                print(
                    f"Step {step_count:3d}: {current_node:2d} -> {action_type} -> {target_node:2d} | VNF: {deployed_vnf}/{ep_stats['vnf_total']}")

            # 执行
            try:
                result = env.step(target_node)
                if len(result) == 5:
                    obs, reward, done, truncated, step_info = result
                else:
                    obs, reward, done, step_info = result
                # 🔥 更新step_info用于下次获取mask
            except Exception as e:
                print(f"❌ Step失败: {e}")
                import traceback
                traceback.print_exc()
                break

            # 检查结果
            if step_info.get('action_type') == 'deploy':
                stats['vnf_deploy_attempts'] += 1
                if step_info.get('success'):
                    stats['vnf_deploy_success'] += 1
                    ep_stats['vnf_deployed'] += 1
                    print(f"   ✅ VNF部署成功！{ep_stats['vnf_deployed']}/{ep_stats['vnf_total']}")
                else:
                    stats['vnf_deploy_failed'] += 1

            if step_info.get('error'):
                stats['errors'][step_info['error']] += 1

            if done and step_info.get('request_completed'):
                ep_stats['success'] = True

        # 统计
        stats['total_episodes'] += 1
        stats['total_steps'] += step_count

        if ep_stats['success']:
            stats['successful_episodes'] += 1
        else:
            stats['failed_episodes'] += 1

        if ep_stats['vnf_deployed'] == 0:
            stats['episodes_no_vnf'] += 1
        elif ep_stats['vnf_deployed'] < ep_stats['vnf_total']:
            stats['episodes_partial_vnf'] += 1
        else:
            stats['episodes_all_vnf'] += 1

        print(
            f"\n📊 摘要: 步数={step_count}, 停留={ep_stats['stay_count']}, VNF={ep_stats['vnf_deployed']}/{ep_stats['vnf_total']}, {'✅成功' if ep_stats['success'] else '❌失败'}")

    # 报告
    print(f"\n{'=' * 80}")
    print(f"📈 诊断报告")
    print(f"{'=' * 80}\n")

    total_actions = stats['action_stay'] + stats['action_move']

    print(
        f"1️⃣ Episode: 成功={stats['successful_episodes']}/{stats['total_episodes']} ({stats['successful_episodes'] / max(1, stats['total_episodes']) * 100:.1f}%)")
    print(
        f"2️⃣ 动作: 停留={stats['action_stay']} ({stats['action_stay'] / max(1, total_actions) * 100:.1f}%), 移动={stats['action_move']} ({stats['action_move'] / max(1, total_actions) * 100:.1f}%)")
    print(
        f"3️⃣ VNF部署: 尝试={stats['vnf_deploy_attempts']}, 成功={stats['vnf_deploy_success']}, 失败={stats['vnf_deploy_failed']}")
    print(
        f"4️⃣ VNF阶段: 0个={stats['episodes_no_vnf']}, 部分={stats['episodes_partial_vnf']}, 全部={stats['episodes_all_vnf']}")

    # 结论
    print(f"\n{'=' * 80}")
    print(f"🎯 结论")
    print(f"{'=' * 80}\n")

    if stats['action_stay'] == 0:
        print("⚠️  严重：从不停留！无法部署VNF！")
    if stats['episodes_no_vnf'] == stats['total_episodes']:
        print("⚠️  严重：所有episode都没部署VNF！")
    if stats['successful_episodes'] == 0:
        print("⚠️  严重：没有成功的episode！")

    if stats['action_stay'] > 0 and stats['vnf_deploy_success'] > 0:
        print("✅ Agent会停留并成功部署VNF")

    print(f"\n{'=' * 80}\n")

    return stats


# ================================================================================
# 主程序
# ================================================================================
if __name__ == "__main__":
    print("🚀 启动诊断工具...\n")

    # 加载配置
    with open('configs/base.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("✅ 配置加载成功\n")

    # 初始化环境
    print("🔧 初始化环境...")
    env = SFC_HIRL_Env(config, use_gnn=True)
    print("✅ 环境初始化成功\n")

    # 初始化Agent
    print("🔧 初始化Agent...")
    agent = HRLAgent(config, device='cpu')
    print("✅ Agent初始化成功\n")

    # 加载数据
    print("📂 加载Phase3数据...")
    with open('data/input_dir/generate_requests_depend_on_poisson/data/到达率50/phase3_requests.pkl', 'rb') as f:
        requests = pickle.load(f)
    with open('data/input_dir/generate_requests_depend_on_poisson/data/到达率50/phase3_requests_by_slot.pkl', 'rb') as f:
        requests_by_slot = pickle.load(f)

    env.all_requests = requests
    env.load_requests(requests, requests_by_slot)
    print(f"✅ 数据加载成功: {len(requests)} 个请求\n")

    # 运行诊断
    print("=" * 80)
    print("🔬 开始诊断")
    print("=" * 80)

    stats = diagnose_phase3_training(env, agent, num_episodes=10)

    print("\n✅ 诊断完成！")