"""
envs/modules/event_handler.py
✅ 修复: 添加 self.services 属性，防止 AttributeError
"""
import logging

logger = logging.getLogger(__name__)


class EventHandler:
    def __init__(self, resource_manager):
        self.resource_mgr = resource_manager
        # 🔥【关键修复】必须初始化 services 字典
        self.services = {}

    def register_service(self, req_id, deployment_info):
        """注册已部署的服务 (用于后续释放)"""
        self.services[req_id] = deployment_info

    def unregister_service(self, req_id):
        """注销服务并释放资源"""
        if req_id in self.services:
            service = self.services.pop(req_id)
            req = service.get('req')
            tree = service.get('tree')

            if req and tree:
                # 调用资源管理器释放资源
                self.resource_mgr.release_resources_from_req(req, tree)
            return True
        return False

    def process_leaves(self, leave_list):
        """批量处理离开事件"""
        for req_id in leave_list:
            self.unregister_service(req_id)

    def reset(self):
        """重置状态"""
        self.services.clear()