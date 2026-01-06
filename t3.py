#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间管理诊断工具
专门分析为什么Episode会立即超时
"""

import numpy as np
from collections import defaultdict
import yaml
import pickle

from envs.sfc_env3 import SFC_HIRL_Env
from core.hrl.agent import HRLAgent


def diagnose_time_management(env, agent, num_episodes=10):
    """诊断时间管理问题"""
    print("=" * 80)
    print("⏰ 时间管理诊断工具")
    print("=" * 80)

    stats = {
        'episodes': [],
        'immediate_timeouts': 0,
        'normal_episodes': 0,
    }

    print(f"\n开始诊断 {num_episodes} 个episodes...\n")

    for ep in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        # ========================================
        # 1. 记录Reset前的状态
        # ========================================
        before_reset = {
            'time_step': getattr(env, 'time_step', None),
            'current_time_slot': getattr(env, 'current_time_slot', None),
            'global_request_index': getattr(env, 'global_request_index', None),
        }

        print(f"\n📊 Reset前状态:")
        print(f"   time_step: {before_reset['time_step']}")
        print(f"   current_time_slot: {before_reset['current_time_slot']}")
        print(f"   global_request_index: {before_reset['global_request_index']}")

        # ========================================
        # 2. 执行Reset
        # ========================================
        obs, info = env.reset()

        # ========================================
        # 3. 记录Reset后的状态
        # ========================================
        after_reset = {
            'time_step': getattr(env, 'time_step', None),
            'current_time_slot': getattr(env, 'current_time_slot', None),
            'global_request_index': getattr(env, 'global_request_index', None),
        }

        print(f"\n📊 Reset后状态:")
        print(f"   time_step: {after_reset['time_step']}")
        print(f"   current_time_slot: {after_reset['current_time_slot']}")
        print(f"   global_request_index: {after_reset['global_request_index']}")

        # ========================================
        # 4. 检查当前请求
        # ========================================
        req = env.current_request

        if req is None:
            print(f"\n❌ 没有请求！")
            stats['immediate_timeouts'] += 1
            continue

        req_info = {
            'id': req.get('id', '?'),
            'source': req.get('source', '?'),
            'vnf': req.get('vnf', []),
            'dest': req.get('dest', []),
            'arrival_time': req.get('arrival_time', None),
            'lifetime': req.get('lifetime', None),
            'time_slot': req.get('time_slot', None),
        }

        print(f"\n📋 当前请求:")
        print(f"   ID: {req_info['id']}")
        print(f"   Arrival Time: {req_info['arrival_time']}")
        print(f"   Lifetime: {req_info['lifetime']}")
        print(f"   Time Slot: {req_info['time_slot']}")
        print(f"   VNF链: {req_info['vnf']} (共{len(req_info['vnf'])}个)")
        print(f"   目的地: {req_info['dest']} (共{len(req_info['dest'])}个)")

        # ========================================
        # 5. 计算请求的过期时间
        # ========================================
        if req_info['arrival_time'] is not None and req_info['lifetime'] is not None:
            expire_time = req_info['arrival_time'] + req_info['lifetime']
            current_time = after_reset['time_step']
            remaining_time = expire_time - current_time if current_time is not None else None

            print(f"\n⏱️ 时间分析:")
            print(f"   当前时间: {current_time}")
            print(f"   过期时间: {expire_time}")
            print(f"   剩余时间: {remaining_time}")

            if remaining_time is not None and remaining_time <= 0:
                print(f"   ⚠️  请求已过期！")

        # ========================================
        # 6. 执行第一步动作
        # ========================================
        print(f"\n🎬 执行第一步动作...")

        try:
            action = agent.select_action(obs)

            # 处理HRL动作格式
            if isinstance(action, (tuple, list)):
                target_node = int(action[1]) if len(action) >= 2 else env.current_node_location
            elif isinstance(action, (int, np.integer)):
                target_node = int(action)
            else:
                print(f"❌ 未知动作格式")
                continue

            print(f"   动作: {env.current_node_location} -> {target_node}")

        except Exception as e:
            print(f"❌ 动作选择失败: {e}")
            continue

        # ========================================
        # 7. 执行Step并检查
        # ========================================
        try:
            # 记录step前的时间
            before_step_time = env.time_step
            before_step_slot = env.current_time_slot

            result = env.step(target_node)
            if len(result) == 5:
                obs, reward, done, truncated, step_info = result
            else:
                obs, reward, done, step_info = result

            # 记录step后的时间
            after_step_time = env.time_step
            after_step_slot = env.current_time_slot

            print(f"\n📊 Step执行结果:")
            print(f"   Done: {done}")
            print(f"   Reward: {reward:.2f}")
            print(
                f"   时间变化: {before_step_time} -> {after_step_time} (Δ={after_step_time - before_step_time if before_step_time and after_step_time else '?'})")
            print(f"   时间切片: {before_step_slot} -> {after_step_slot}")

            if done:
                print(f"   Done原因: {step_info.get('error', 'unknown')}")

                # 检查是否是立即超时
                if step_count == 1:
                    print(f"\n⚠️  立即超时！只执行了1步！")
                    stats['immediate_timeouts'] += 1

                    # 详细分析
                    print(f"\n🔍 立即超时分析:")
                    print(f"   1. 请求在Reset时是否已过期？")
                    print(f"      - 到达时间: {req_info['arrival_time']}")
                    print(f"      - 生命周期: {req_info['lifetime']}")
                    print(f"      - 过期时间: {expire_time if 'expire_time' in locals() else '?'}")
                    print(f"      - Reset后时间: {after_reset['time_step']}")

                    print(f"   2. Step是否触发了时间切片切换？")
                    print(f"      - Step前时间切片: {before_step_slot}")
                    print(f"      - Step后时间切片: {after_step_slot}")
                    print(f"      - 是否切换: {'是' if before_step_slot != after_step_slot else '否'}")

                    print(f"   3. 请求是否在active_requests中？")
                    active = getattr(env, 'active_requests', {})
                    print(f"      - active_requests数量: {len(active)}")
                    print(f"      - 请求ID在其中: {req_info['id'] in active}")
                else:
                    stats['normal_episodes'] += 1

        except Exception as e:
            print(f"❌ Step执行失败: {e}")
            import traceback
            traceback.print_exc()

        # 记录episode信息
        ep_data = {
            'episode': ep + 1,
            'before_reset': before_reset,
            'after_reset': after_reset,
            'request': req_info,
            'done_in_one_step': done if 'done' in locals() else None,
        }
        stats['episodes'].append(ep_data)

    # ========================================
    # 📊 统计报告
    # ========================================
    print(f"\n{'=' * 80}")
    print(f"📈 时间管理诊断报告")
    print(f"{'=' * 80}\n")

    print(f"1️⃣ Episode统计:")
    print(f"   总数: {num_episodes}")
    print(f"   立即超时: {stats['immediate_timeouts']} ({stats['immediate_timeouts'] / num_episodes * 100:.1f}%)")
    print(f"   正常执行: {stats['normal_episodes']} ({stats['normal_episodes'] / num_episodes * 100:.1f}%)")

    # 分析立即超时的原因
    if stats['immediate_timeouts'] > 0:
        print(f"\n2️⃣ 立即超时原因分析:")

        timeout_reasons = {
            'request_expired_at_reset': 0,
            'time_slot_switch': 0,
            'other': 0,
        }

        for ep_data in stats['episodes']:
            if ep_data.get('done_in_one_step'):
                req = ep_data['request']
                after = ep_data['after_reset']

                # 检查是否在reset时已过期
                if req['arrival_time'] and req['lifetime'] and after['time_step']:
                    expire_time = req['arrival_time'] + req['lifetime']
                    if after['time_step'] >= expire_time:
                        timeout_reasons['request_expired_at_reset'] += 1
                        continue

                # 其他原因
                timeout_reasons['other'] += 1

        print(f"   Reset时请求已过期: {timeout_reasons['request_expired_at_reset']}")
        print(f"   时间切片切换: {timeout_reasons['time_slot_switch']}")
        print(f"   其他原因: {timeout_reasons['other']}")

    print(f"\n{'=' * 80}")
    print(f"✅ 诊断完成")
    print(f"{'=' * 80}\n")

    return stats


# ================================================================================
# 主程序
# ================================================================================
if __name__ == "__main__":
    print("🚀 启动时间管理诊断...\n")

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
    stats = diagnose_time_management(env, agent, num_episodes=10)