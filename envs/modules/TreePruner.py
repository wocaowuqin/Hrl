from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class TreePruner:
    """
    🔥 全能树剪枝器 V25.0 MAB增强版

    功能：
    1. MAB智能剪枝（可选）
    2. 反向回溯Essential边识别
    3. 树连通性验证
    4. 资源预留管理
    """

    def __init__(self, resource_mgr, config=None):
        """
        初始化剪枝器

        Args:
            resource_mgr: 资源管理器
            config: 配置字典，包含：
                - use_mab_pruning: 是否启用MAB剪枝
                - mab_rounds: MAB探索轮数
                - enable_mab_learning: 是否启用MAB学习
        """
        self.resource_mgr = resource_mgr
        self.config = config or {}

        # MAB相关配置
        self.use_mab_pruning = self.config.get('use_mab_pruning', False)
        self.mab_rounds = self.config.get('mab_rounds', 10)
        self.enable_mab_learning = self.config.get('enable_mab_learning', False)

        # MAB统计
        self.mab_action_stats = {
            'total_selections': 0,
            'positive_rewards': 0,
            'negative_rewards': 0,
            'successful_prunes': 0,
            'failed_prunes': 0
        }

        # 当前状态
        self.current_request = None
        self.current_tree = None

        self._init_Tree_Pruner(resource_mgr, config)

    def _init_Tree_Pruner(self, resource_mgr, config=None):
        """
        🔥 修改点 2 [新增方法]：构建 TreePruner 的内部状态和配置

        Args:
            resource_mgr: 资源管理器实例
            config: 配置字典
        """
        # 1. 基础依赖注入
        self.resource_mgr = resource_mgr
        self.config = config or {}

        # 2. 解析 MAB (Multi-Armed Bandit) 相关配置
        # 提供默认值以防 config 为空或缺失键
        self.use_mab_pruning = self.config.get('use_mab_pruning', False)
        self.mab_rounds = self.config.get('mab_rounds', 10)
        self.enable_mab_learning = self.config.get('enable_mab_learning', False)

        # 3. 初始化 MAB 统计数据结构
        # 用于跟踪剪枝决策的效果
        self.mab_action_stats = {
            'total_selections': 0,  # 总共尝试剪枝次数
            'positive_rewards': 0,  # 获得正奖励次数 (剪枝成功且有效)
            'negative_rewards': 0,  # 获得负奖励次数 (剪枝导致断连或性能下降)
            'successful_prunes': 0,  # 成功执行的剪枝操作数
            'failed_prunes': 0  # 失败的剪枝操作数
        }

        # 4. 初始化上下文状态容器
        # 这些将在 set_current_request 中被填充
        self.current_request = None  # 当前处理的 SFC 请求
        self.current_tree = None  # 当前构建的多播树结构

        # 5. MAB 算法实例占位符
        # 需要通过 set_mab_pruner 单独注入
        self.mab_pruner = None

        logger.debug(f"TreePruner 初始化完成 | MAB启用: {self.use_mab_pruning}")
    def set_current_request(self, request, tree):
        """设置当前请求和树"""
        self.current_request = request
        self.current_tree = tree

    def prune(self):
        """
        🔥 主剪枝入口

        Returns:
            tuple: (pruned_tree, valid_nodes, success, extra_info)
        """
        if not self.current_request:
            logger.error("未设置当前请求")
            return {}, set(), False, {}

        if self.use_mab_pruning and hasattr(self, 'mab_pruner'):
            return self._prune_with_mab()
        else:
            return self._prune_without_mab()

    def _prune_without_mab(self):
        """
        传统剪枝：仅保留Essential边

        Returns:
            tuple: (final_tree_edges, valid_nodes, True, parent_map)
        """
        # 基础检查
        if not self.current_request:
            return {}, set(), False, {}

        req = self.current_request
        source = req.get('source')
        dests = set(req.get('dest', []))
        placement = self.current_tree.get('placement', {})
        current_tree_edges = self.current_tree.get('tree', {})

        if not current_tree_edges:
            return {}, {source}, False, {}

        # Phase 1: 识别Essential Edges & 构建Parent Map
        adj = defaultdict(list)
        for u, v in current_tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        # BFS构建父节点映射
        parent_map = {source: None}
        queue = deque([source])
        visited = {source}

        while queue:
            curr = queue.popleft()
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = curr
                    queue.append(neighbor)

        # 识别关键节点
        critical_nodes = dests.copy()
        for key in placement.keys():
            if isinstance(key, tuple):
                critical_nodes.add(key[0])

        # 反向回溯Essential Edges
        essential_edges = set()
        valid_nodes = {source}

        for node in critical_nodes:
            curr = node
            if curr not in visited:
                continue

            valid_nodes.add(curr)
            while curr != source and curr in parent_map:
                p = parent_map[curr]
                if p is None:
                    break
                # 规范化边
                edge = tuple(sorted((p, curr)))
                essential_edges.add(edge)
                valid_nodes.add(p)
                curr = p

        # 构建最终树
        final_tree_edges = {}
        for (u, v), data in current_tree_edges.items():
            edge_key = tuple(sorted((u, v)))
            if edge_key in essential_edges:
                final_tree_edges[(u, v)] = data

        return final_tree_edges, valid_nodes, True, parent_map

    def _prune_with_mab(self):
        """
        🔥 MAB增强版剪枝

        Returns:
            tuple: (pruned_tree, valid_nodes, prune_success, mab_info)
        """
        if not self.current_request:
            return {}, set(), False, {}

        req = self.current_request
        source = req.get('source')
        dests = set(req.get('dest', []))
        vnf_list = req.get('vnf', [])
        placement = self.current_tree.get('placement', {})
        current_tree_edges = self.current_tree.get('tree', {})
        bw_req = req.get('bw_origin', 1.0)

        if not current_tree_edges:
            return {}, {source}, False, {}

        # ---------------------------------------------------------
        # Phase 1: 识别Essential Edges (基准线)
        # ---------------------------------------------------------
        logger.debug(f"Phase 1: 识别Essential Edges, 源: {source}, 目的: {dests}")

        # 构建邻接表
        adj = defaultdict(list)
        for u, v in current_tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        # BFS构建父节点映射
        parent = {source: None}
        queue = deque([source])
        visited = {source}
        while queue:
            curr = queue.popleft()
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

        # 识别关键节点 (Dest + VNF放置节点)
        critical_nodes = dests.copy()
        for key in placement.keys():
            if isinstance(key, tuple):
                critical_nodes.add(key[0])

        logger.debug(f"关键节点集合: {critical_nodes}")

        # 反向回溯标记Essential Edges
        essential_edges = set()
        valid_nodes = {source}  # 基础有效节点

        for node in critical_nodes:
            curr = node
            # 如果关键节点不可达，说明树本身断了
            if curr not in visited:
                logger.warning(f"关键节点 {curr} 不可达，树可能不连通")
                continue

            valid_nodes.add(curr)
            while curr != source and curr in parent:
                p = parent[curr]
                if p is None:
                    break
                edge = self.mab_pruner._normalize_edge((p, curr))
                essential_edges.add(edge)
                valid_nodes.add(p)
                curr = p

        logger.debug(f"Phase 1完成: Essential Edges={len(essential_edges)}, Valid Nodes={len(valid_nodes)}")

        # ---------------------------------------------------------
        # Phase 2: MAB动态评估 (探索非Essential边)
        # ---------------------------------------------------------
        if not self.use_mab_pruning:
            # 如果未开启MAB，直接返回Essential Tree
            pruned_tree = {}
            for (u, v), data in current_tree_edges.items():
                edge_key = self.mab_pruner._normalize_edge((u, v))
                if edge_key in essential_edges:
                    pruned_tree[(u, v)] = data

            logger.debug("MAB剪枝未启用，使用传统反向回溯")
            return pruned_tree, valid_nodes, True, {
                'method': 'backward_only',
                'essential_edges': len(essential_edges),
                'total_edges': len(current_tree_edges)
            }

        # 候选边 = 所有边 - Essential Edges
        all_edges_set = set(self.mab_pruner._normalize_edge(e) for e in current_tree_edges.keys())
        candidate_edges = all_edges_set - essential_edges

        logger.debug(f"候选边数量: {len(candidate_edges)} (总数: {len(all_edges_set)}, 关键: {len(essential_edges)})")

        if not candidate_edges:
            # 没有可优化的余地
            pruned_tree = {}
            for (u, v), data in current_tree_edges.items():
                edge_key = self.mab_pruner._normalize_edge((u, v))
                if edge_key in essential_edges:
                    pruned_tree[(u, v)] = data

            logger.debug("无候选边可优化")
            return pruned_tree, valid_nodes, True, {
                'method': 'backward_only',
                'candidates': 0,
                'essential_edges': len(essential_edges)
            }

        # 初始化MAB统计
        self.mab_pruner.initialize_edges(candidate_edges)

        # 构建原始树的副本用于MAB探索
        original_tree = current_tree_edges.copy()

        # MAB探索: 尝试剪除候选边
        edges_to_remove = set()
        edges_to_keep = set(candidate_edges)  # 初始假设所有候选边都保留

        for round_idx in range(self.mab_rounds):
            if not edges_to_keep:
                logger.debug(f"第{round_idx}轮: 无更多候选边可探索")
                break

            # MAB选择一条边尝试剪除
            selected_edge = self.mab_pruner.select_edge(
                {self.mab_pruner._normalize_edge(e) for e in edges_to_keep},
                self.mab_action_stats['total_selections']
            )

            if not selected_edge:
                logger.debug(f"第{round_idx}轮: MAB未选择边")
                break

            self.mab_action_stats['total_selections'] += 1

            # 检查这条边是否仍然在候选集合中
            if selected_edge not in edges_to_keep:
                logger.debug(f"第{round_idx}轮: 边{selected_edge}已不在候选集合中")
                continue

            # 模拟剪除这条边
            # 构建剪除后的树
            temp_tree = {}
            for (u, v), data in original_tree.items():
                edge_key = self.mab_pruner._normalize_edge((u, v))
                # 保留所有essential边和未选中的候选边
                if edge_key in essential_edges or (edge_key in candidate_edges and edge_key != selected_edge):
                    temp_tree[(u, v)] = data

            # 验证剪除后的树是否仍然连通
            is_connected = self._verify_tree_connectivity(temp_tree, source, critical_nodes)

            # 计算奖励
            reward = self.mab_pruner.compute_reward(
                tree_before=original_tree,
                tree_after=temp_tree,
                bw_req=bw_req,
                constraints_satisfied=is_connected,
                network_utilization=self.resource_mgr.get_network_utilization() if hasattr(self.resource_mgr,
                                                                                           'get_network_utilization') else 0.5
            )

            # 更新MAB统计
            if self.enable_mab_learning:
                self.mab_pruner.update_edge_reward(
                    selected_edge,
                    reward,
                    self.mab_action_stats['total_selections']
                )

            # 更新MAB动作统计
            if reward > 0:
                self.mab_action_stats['positive_rewards'] += 1
                self.mab_action_stats['successful_prunes'] += 1
                edges_to_remove.add(selected_edge)
                edges_to_keep.remove(selected_edge)
                logger.debug(f"第{round_idx}轮: 剪除边{selected_edge}, 奖励: {reward:.3f} (成功)")
            else:
                self.mab_action_stats['negative_rewards'] += 1
                self.mab_action_stats['failed_prunes'] += 1
                # 负奖励时保留该边
                logger.debug(f"第{round_idx}轮: 保留边{selected_edge}, 奖励: {reward:.3f} (失败)")

        # ---------------------------------------------------------
        # Phase 3: 生成最终树
        # ---------------------------------------------------------
        final_tree_edges = {}
        for (u, v), data in current_tree_edges.items():
            edge_key = self.mab_pruner._normalize_edge((u, v))

            # Essential边必须保留
            if edge_key in essential_edges:
                final_tree_edges[(u, v)] = data
                valid_nodes.add(u)
                valid_nodes.add(v)
            # 候选边根据MAB决定
            elif edge_key in candidate_edges:
                if edge_key in edges_to_remove:
                    # MAB决定剪除
                    logger.debug(f"剪除候选边: {edge_key}")
                else:
                    # MAB决定保留或未探索
                    final_tree_edges[(u, v)] = data
                    valid_nodes.add(u)
                    valid_nodes.add(v)
                    logger.debug(f"保留候选边: {edge_key}")
            else:
                # 其他边(不应该出现)
                logger.warning(f"发现未分类的边: {edge_key}")

        logger.info(f"MAB剪枝完成: 原始边={len(current_tree_edges)}, "
                    f"最终边={len(final_tree_edges)}, "
                    f"剪除={len(edges_to_remove)}")

        return final_tree_edges, valid_nodes, True, {
            'method': 'mab_enhanced',
            'removed': len(edges_to_remove),
            'candidates': len(candidate_edges),
            'essential_edges': len(essential_edges),
            'total_edges': len(current_tree_edges),
            'final_edges': len(final_tree_edges),
            'mab_stats': self.mab_action_stats.copy()
        }

    def _verify_tree_connectivity(self, tree_edges, source, critical_nodes):
        """
        验证树是否连通所有关键节点

        Args:
            tree_edges: 树的边集合
            source: 源节点
            critical_nodes: 关键节点集合

        Returns:
            bool: 是否连通
        """
        if not tree_edges:
            return False

        # 构建邻接表
        adj = defaultdict(list)
        for u, v in tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        # BFS遍历
        visited = set()
        queue = deque([source])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        # 检查所有关键节点是否都被访问到
        for node in critical_nodes:
            if node not in visited:
                return False

        return True

    def try_reserve_resources(self, tx_id, placement, tree_edges, valid_nodes=None):
        """
        尝试预留资源 - 修复版

        修复要点：
        1. 兼容多种placement key格式（2元组、3元组等）
        2. 从info字典中提取node和vnf信息（更可靠）
        3. 添加详细的错误日志

        Args:
            tx_id: 事务ID
            placement: VNF放置信息 {key: info}
            tree_edges: 树边集合 {(u,v): bw}
            valid_nodes: 有效节点集合（可选）

        Returns:
            bool: 资源预留是否成功
        """
        # 1. 构建有效节点集合
        if valid_nodes is None:
            valid_nodes = set()
            for (u, v) in tree_edges.keys():
                valid_nodes.add(u)
                valid_nodes.add(v)

        req = self.current_request
        bw = req.get('bw_origin', 1.0)

        # 2. 预留节点资源 (VNF放置)
        reserved_nodes = []

        for key, info in placement.items():
            # 🔥 修复点1：兼容多种key格式
            # 优先从info字典中提取信息
            if isinstance(info, dict):
                node_id = info.get('node')
                vnf_type = info.get('vnf_type')

                # 如果info中没有，尝试从key中提取
                if node_id is None or vnf_type is None:
                    if isinstance(key, tuple):
                        if len(key) >= 2:
                            node_id = key[0]
                            vnf_type = key[1]
                        else:
                            logger.warning(f"⚠️ placement key格式异常: {key}, 跳过")
                            continue
                    else:
                        logger.warning(f"⚠️ placement key不是tuple: {key}, 跳过")
                        continue
            else:
                # info不是字典，尝试从key中提取
                if isinstance(key, tuple) and len(key) >= 2:
                    node_id = key[0]
                    vnf_type = key[1]
                else:
                    logger.warning(f"⚠️ 无法解析placement: key={key}, info={info}")
                    continue

            # 检查节点是否有效
            if node_id not in valid_nodes:
                logger.debug(f"节点 {node_id} 不在有效节点集合中，跳过")
                continue

            # 🔥 修复点2：获取资源需求
            # 优先从info中获取
            if isinstance(info, dict):
                cpu_needed = info.get('cpu_used', 1.0)
                mem_needed = info.get('mem_used', 1.0)
            else:
                # 回退到从请求中获取
                vnf_list = req.get('vnf', [])
                cpu_list = req.get('cpu_origin', [])
                mem_list = req.get('memory_origin', [])

                # 尝试从vnf_type索引获取
                if isinstance(vnf_type, int) and vnf_type < len(cpu_list):
                    cpu_needed = cpu_list[vnf_type]
                    mem_needed = mem_list[vnf_type] if vnf_type < len(mem_list) else 1.0
                else:
                    logger.warning(f"⚠️ 无法获取VNF资源需求，使用默认值")
                    cpu_needed = 1.0
                    mem_needed = 1.0

            # 预留资源
            logger.debug(f"预留节点资源: node={node_id}, vnf={vnf_type}, "
                         f"cpu={cpu_needed:.1f}, mem={mem_needed:.1f}")

            if not self.resource_mgr.reserve_node_resource(
                    tx_id, node_id, vnf_type, cpu_needed, mem_needed
            ):
                logger.warning(f"❌ 节点资源预留失败: node={node_id}, vnf={vnf_type}")
                return False

            reserved_nodes.append((node_id, vnf_type, cpu_needed, mem_needed))

        logger.info(f"✅ 节点资源预留成功: {len(reserved_nodes)} 个VNF")

        # 3. 预留链路资源
        reserved_links = []

        for (u, v) in tree_edges.keys():
            logger.debug(f"预留链路资源: {u}-{v}, bw={bw:.1f}")

            if not self.resource_mgr.reserve_link_resource(tx_id, u, v, bw):
                logger.warning(f"❌ 链路资源预留失败: {u}-{v}")
                return False

            reserved_links.append((u, v, bw))

        logger.info(f"✅ 链路资源预留成功: {len(reserved_links)} 条边")

        # 4. 成功
        logger.info(f"🎉 所有资源预留成功: 节点={len(reserved_nodes)}, 链路={len(reserved_links)}")
        return True

    def finalize_request(self):
        """
        🔥 结算请求 (MAB集成版) 增强错误处理
        """
        if self.current_request is None:
            return False

        req_id = self.current_request.get('id', 'unknown')
        logger.info(f"开始结算请求 {req_id} (MAB剪枝模式)")

        # 1. 释放当前持有的所有物理资源
        current_tree_edges = self.current_tree.get('tree', {})
        current_placement = self.current_tree.get('placement', {})
        bw = self.current_request.get('bw_origin', 1.0)

        # 释放链路资源
        for (u, v) in current_tree_edges.keys():
            self.resource_mgr.release_link_resource(u, v, bw)

        # 释放节点资源
        for key, info in current_placement.items():
            try:
                # 🔥 兼容多种格式
                if isinstance(info, dict):
                    node = info.get('node', key[0] if isinstance(key, tuple) else None)
                    vnf = info.get('vnf_type', key[1] if isinstance(key, tuple) and len(key) >= 2 else 0)
                    c = info.get('cpu_used', 1.0)
                    m = info.get('mem_used', 1.0)
                else:
                    if isinstance(key, tuple) and len(key) >= 2:
                        node, vnf = key[0], key[1]
                        c, m = 1.0, 1.0
                    else:
                        logger.warning(f"⚠️ 无法解析placement key: {key}")
                        continue

                self.resource_mgr.release_node_resource(node, vnf, c, m)
            except Exception as e:
                logger.error(f"❌ 释放节点资源失败: key={key}, error={e}")
                continue

        logger.info(f"♻️ [结算中间态] 释放暂存资源，准备重组 (MAB模式: {self.use_mab_pruning})")

        # 2. 调用MAB剪枝
        try:
            pruned_tree, valid_nodes, prune_success, mab_info = self.prune()
        except Exception as e:
            logger.error(f"❌ MAB剪枝异常: {e}")
            import traceback
            traceback.print_exc()

            # 回退到传统方法
            logger.warning("⚠️ 回退到传统剪枝方法")
            pruned_tree, valid_nodes, prune_success, parent_map = self._prune_without_mab()
            mab_info = {'method': 'backward_only', 'error': str(e)}

        logger.info(f"🤖 [MAB剪枝] 方法: {mab_info.get('method')}, "
                    f"候选边: {mab_info.get('candidates', 0)}, "
                    f"剪除: {mab_info.get('removed', 0)}, "
                    f"最终边: {mab_info.get('final_edges', 0)}")

        # 打印MAB统计
        if 'mab_stats' in mab_info:
            stats = mab_info['mab_stats']
            logger.info(f"MAB统计: 选择={stats['total_selections']}, "
                        f"正奖励={stats['positive_rewards']}, "
                        f"负奖励={stats['negative_rewards']}")

        # 3. 开始资源预留事务
        tx_id = self.resource_mgr.begin_transaction(req_id)
        final_tree = None

        try:
            plan_success = False

            # 尝试Plan A (剪枝后的树)
            if prune_success:
                try:
                    logger.info("尝试Plan A (剪枝方案)...")

                    # 🔥 详细日志
                    logger.debug(f"Plan A参数: placement keys={list(current_placement.keys())[:3]}..., "
                                 f"tree_edges={len(pruned_tree)}, valid_nodes={len(valid_nodes)}")

                    if self.try_reserve_resources(tx_id, current_placement, pruned_tree, valid_nodes):
                        final_tree = pruned_tree
                        plan_success = True
                        logger.info(f"✅ Plan A (剪枝方案) 资源预留成功")
                    else:
                        logger.warning(f"⚠️ Plan A (剪枝方案) 资源预留失败")
                except Exception as e:
                    logger.error(f"❌ Plan A失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

                    self.resource_mgr.rollback_transaction(tx_id)
                    tx_id = self.resource_mgr.begin_transaction(req_id)

            # 尝试Plan B (回退到原始树)
            if not plan_success:
                logger.warning(f"⚠️ [结算] 剪枝方案不可行，回退原始方案")

                try:
                    logger.info("尝试Plan B (原始方案)...")

                    # 🔥 对于Plan B，使用完整的节点集合
                    original_valid_nodes = set()
                    for (u, v) in current_tree_edges.keys():
                        original_valid_nodes.add(u)
                        original_valid_nodes.add(v)

                    if self.try_reserve_resources(tx_id, current_placement, current_tree_edges, original_valid_nodes):
                        final_tree = current_tree_edges
                        logger.info(f"✅ Plan B (原始方案) 资源预留成功")
                    else:
                        logger.error(f"❌ Plan B (原始方案) 资源预留失败")
                        raise Exception("原始资源无法回收 (可能并发冲突?)")
                except Exception as e:
                    logger.error(f"❌ Plan B失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise

            # 4. 提交事务
            if self.resource_mgr.commit_transaction(tx_id):
                self.current_tree['tree'] = final_tree
                logger.info(f"✅ [结算完成] 请求 {req_id} 成功")

                # 打印MAB总结统计（可选）
                if self.use_mab_pruning and self.enable_mab_learning and hasattr(self, 'mab_pruner'):
                    self.mab_pruner.print_stats()

                return True
            else:
                logger.error(f"❌ [结算] 事务提交失败")
                return False

        except Exception as e:
            logger.error(f"❌ [结算崩溃] {e}")
            import traceback
            logger.error(traceback.format_exc())

            self.resource_mgr.rollback_transaction(tx_id)
            return False

    def debug_print_placement(self, placement):
        """
        打印placement结构用于调试
        """
        logger.info(f"📋 Placement结构调试:")
        logger.info(f"  总数: {len(placement)}")

        for i, (key, info) in enumerate(list(placement.items())[:5]):  # 只打印前5个
            logger.info(
                f"  [{i}] key={key} (type={type(key).__name__}, len={len(key) if isinstance(key, tuple) else 'N/A'})")
            logger.info(f"      info={info}")

        if len(placement) > 5:
            logger.info(f"  ... 还有 {len(placement) - 5} 个")

    def reset_mab_stats(self):
        """重置MAB统计"""
        self.mab_action_stats = {
            'total_selections': 0,
            'positive_rewards': 0,
            'negative_rewards': 0,
            'successful_prunes': 0,
            'failed_prunes': 0
        }

    def set_mab_pruner(self, mab_pruner):
        """设置MAB剪枝器"""
        self.mab_pruner = mab_pruner