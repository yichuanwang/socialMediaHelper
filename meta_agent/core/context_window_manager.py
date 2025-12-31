"""
ContextWindowManager - 上下文窗口管理器
实现原子化加载、动态卸载和 Token 优化
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LifecycleStatus(str, Enum):
    """生命周期状态"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class ContextItem:
    """上下文项"""
    id: str
    content: str
    lifecycle: LifecycleStatus
    task_id: str
    injected_at: str
    metadata: Dict[str, Any]
    token_count: int = 0  # 粗略估计的 token 数


class ContextWindowManager:
    """
    上下文窗口管理器
    
    核心功能：
    1. 原子化加载：标记注入的 Summary 为临时内容
    2. 动态卸载：自动将完成任务的 Summary 从活跃区移除
    3. Token 优化：实时监控并压缩上下文
    """
    
    def __init__(self, max_tokens: int = 8000, max_active_items: int = 10):
        """
        初始化上下文窗口管理器
        
        Args:
            max_tokens: 最大 token 数
            max_active_items: 最大活跃项数量
        """
        self.max_tokens = max_tokens
        self.max_active_items = max_active_items
        
        # 上下文存储
        self._context_items: Dict[str, ContextItem] = {}
        self._active_order: List[str] = []  # 活跃项的顺序（用于LRU）
        
        # 统计信息
        self._total_injections = 0
        self._total_completions = 0
        self._total_archives = 0
        
        logger.debug(f"ContextWindowManager initialized (max_tokens={max_tokens}, max_active_items={max_active_items})")
    
    def inject_content(
        self,
        content: str,
        task_id: str,
        lifecycle_tag: LifecycleStatus = LifecycleStatus.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        注入内容到上下文
        
        Args:
            content: 要注入的内容
            task_id: 任务ID
            lifecycle_tag: 生命周期标签
            metadata: 元数据
            
        Returns:
            上下文项ID
        """
        # 生成唯一ID
        item_id = f"ctx_{task_id}_{datetime.now().timestamp()}"
        
        # 估算 token 数（粗略：每个字符约 0.5 个 token）
        token_count = int(len(content) * 0.5)
        
        # 创建上下文项
        context_item = ContextItem(
            id=item_id,
            content=content,
            lifecycle=lifecycle_tag,
            task_id=task_id,
            injected_at=datetime.now().isoformat(),
            metadata=metadata or {},
            token_count=token_count
        )
        
        # 存储
        self._context_items[item_id] = context_item
        
        # 如果是活跃状态，添加到活跃列表
        if lifecycle_tag == LifecycleStatus.ACTIVE:
            self._active_order.append(item_id)
            self._total_injections += 1
        
        logger.debug(f"Content injected: {item_id} (task={task_id}, tokens={token_count}, lifecycle={lifecycle_tag})")
        
        # 检查是否需要压缩
        self._check_and_compress()
        
        return item_id
    
    def mark_as_completed(self, task_id: str) -> bool:
        """
        标记任务为已完成
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功
        """
        completed_count = 0
        
        for item_id, item in self._context_items.items():
            if item.task_id == task_id and item.lifecycle == LifecycleStatus.ACTIVE:
                # 更新状态
                item.lifecycle = LifecycleStatus.COMPLETED
                
                # 从活跃列表移除
                if item_id in self._active_order:
                    self._active_order.remove(item_id)
                
                completed_count += 1
                logger.debug(f"Context item marked as completed: {item_id}")
        
        if completed_count > 0:
            self._total_completions += completed_count
            # 触发压缩
            self._compress_completed_items()
            return True
        
        return False
    
    def compress_and_archive(self, item_id: str) -> Optional[str]:
        """
        压缩并归档内容
        
        Args:
            item_id: 上下文项ID
            
        Returns:
            压缩后的摘要
        """
        if item_id not in self._context_items:
            return None
        
        item = self._context_items[item_id]
        
        # 生成压缩摘要（保留前100个字符）
        summary = item.content[:100] + "..." if len(item.content) > 100 else item.content
        
        # 更新状态
        item.lifecycle = LifecycleStatus.ARCHIVED
        item.content = summary  # 替换为摘要
        item.token_count = int(len(summary) * 0.5)
        
        self._total_archives += 1
        logger.debug(f"Context item archived: {item_id} (compressed to {len(summary)} chars)")
        
        return summary
    
    def get_active_context(self) -> List[Dict[str, Any]]:
        """
        获取活跃的上下文
        
        Returns:
            活跃上下文列表
        """
        active_items = []
        
        for item_id in self._active_order:
            if item_id in self._context_items:
                item = self._context_items[item_id]
                active_items.append({
                    "role": "developer",
                    "content": item.content,
                    "metadata": {
                        **item.metadata,
                        "lifecycle": item.lifecycle.value,
                        "task_id": item.task_id,
                        "injected_at": item.injected_at
                    }
                })
        
        return active_items
    
    def get_all_context(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        """
        获取所有上下文
        
        Args:
            include_archived: 是否包含已归档的项
            
        Returns:
            上下文列表
        """
        context_list = []
        
        for item in self._context_items.values():
            if not include_archived and item.lifecycle == LifecycleStatus.ARCHIVED:
                continue
            
            context_list.append({
                "role": "developer",
                "content": item.content,
                "metadata": {
                    **item.metadata,
                    "lifecycle": item.lifecycle.value,
                    "task_id": item.task_id,
                    "injected_at": item.injected_at
                }
            })
        
        return context_list
    
    def _check_and_compress(self):
        """检查并压缩上下文"""
        # 检查活跃项数量
        if len(self._active_order) > self.max_active_items:
            logger.debug(f"Active items exceed limit ({len(self._active_order)} > {self.max_active_items}), compressing...")
            self._compress_by_count()
        
        # 检查 token 数量
        total_tokens = sum(
            item.token_count for item in self._context_items.values()
            if item.lifecycle in [LifecycleStatus.ACTIVE, LifecycleStatus.COMPLETED]
        )
        
        if total_tokens > self.max_tokens:
            logger.debug(f"Total tokens exceed limit ({total_tokens} > {self.max_tokens}), compressing...")
            self._compress_by_tokens()
    
    def _compress_by_count(self):
        """按数量压缩（移除最旧的活跃项）"""
        while len(self._active_order) > self.max_active_items:
            # 移除最旧的项
            oldest_id = self._active_order.pop(0)
            
            if oldest_id in self._context_items:
                item = self._context_items[oldest_id]
                # 标记为已完成（如果还是活跃状态）
                if item.lifecycle == LifecycleStatus.ACTIVE:
                    item.lifecycle = LifecycleStatus.COMPLETED
                
                # 归档
                self.compress_and_archive(oldest_id)
    
    def _compress_by_tokens(self):
        """按 token 数压缩"""
        # 找出所有已完成的项并归档
        completed_items = [
            item_id for item_id, item in self._context_items.items()
            if item.lifecycle == LifecycleStatus.COMPLETED
        ]
        
        for item_id in completed_items:
            self.compress_and_archive(item_id)
        
        # 如果还是超过限制，压缩最旧的活跃项
        total_tokens = sum(
            item.token_count for item in self._context_items.values()
            if item.lifecycle == LifecycleStatus.ACTIVE
        )
        
        while total_tokens > self.max_tokens and self._active_order:
            oldest_id = self._active_order.pop(0)
            
            if oldest_id in self._context_items:
                item = self._context_items[oldest_id]
                old_tokens = item.token_count
                
                # 归档
                self.compress_and_archive(oldest_id)
                
                # 更新总 token 数
                total_tokens -= old_tokens
                total_tokens += item.token_count  # 加上压缩后的 token 数
    
    def _compress_completed_items(self):
        """压缩已完成的项"""
        # 保留最近 3 个已完成的任务
        completed_items = [
            (item_id, item) for item_id, item in self._context_items.items()
            if item.lifecycle == LifecycleStatus.COMPLETED
        ]
        
        # 按注入时间排序
        completed_items.sort(key=lambda x: x[1].injected_at, reverse=True)
        
        # 归档超过 3 个的项
        for item_id, item in completed_items[3:]:
            self.compress_and_archive(item_id)
    
    def clear_archived(self):
        """清除已归档的项"""
        archived_ids = [
            item_id for item_id, item in self._context_items.items()
            if item.lifecycle == LifecycleStatus.ARCHIVED
        ]
        
        for item_id in archived_ids:
            del self._context_items[item_id]
        
        logger.debug(f"Cleared {len(archived_ids)} archived items")
    
    def clear_all(self):
        """清除所有上下文"""
        self._context_items.clear()
        self._active_order.clear()
        logger.debug("All context cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        # 按状态统计
        status_counts = {
            "pending": 0,
            "active": 0,
            "completed": 0,
            "archived": 0
        }
        
        total_tokens = 0
        
        for item in self._context_items.values():
            status_counts[item.lifecycle.value] += 1
            if item.lifecycle in [LifecycleStatus.ACTIVE, LifecycleStatus.COMPLETED]:
                total_tokens += item.token_count
        
        return {
            "total_items": len(self._context_items),
            "active_items": len(self._active_order),
            "status_counts": status_counts,
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "token_usage_rate": f"{(total_tokens / self.max_tokens * 100):.1f}%",
            "total_injections": self._total_injections,
            "total_completions": self._total_completions,
            "total_archives": self._total_archives
        }
    
    def get_item_by_task_id(self, task_id: str) -> List[ContextItem]:
        """
        根据任务ID获取上下文项
        
        Args:
            task_id: 任务ID
            
        Returns:
            上下文项列表
        """
        return [
            item for item in self._context_items.values()
            if item.task_id == task_id
        ]
