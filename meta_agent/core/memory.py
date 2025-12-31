"""
Memory模块 - 管理Agent的对话历史和记忆
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class Message:
    """消息类"""
    
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化消息
        
        Args:
            role: 角色（user/assistant/system）
            content: 消息内容
            timestamp: 时间戳
            metadata: 元数据
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建"""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )


class Memory:
    """
    Memory类 - 管理对话历史和重要信息
    
    功能：
    1. 短期记忆：存储最近的对话历史
    2. 长期记忆：存储重要的上下文信息
    3. 上下文窗口管理：防止token溢出
    """
    
    def __init__(
        self,
        session_id: str,
        max_short_term_messages: int = 20,
        max_tokens: int = 4000
    ):
        """
        初始化Memory
        
        Args:
            session_id: 会话ID
            max_short_term_messages: 短期记忆最大消息数
            max_tokens: 最大token数（用于上下文窗口管理）
        """
        self.session_id = session_id
        self.max_short_term_messages = max_short_term_messages
        self.max_tokens = max_tokens
        
        # 短期记忆：对话历史
        self.short_term_memory: List[Message] = []
        
        # 长期记忆：重要信息
        self.long_term_memory: Dict[str, Any] = {
            "user_preferences": {},  # 用户偏好
            "task_context": {},      # 任务上下文
            "important_facts": [],   # 重要事实
            "generated_content": []  # 已生成的内容记录
        }
        
        logger.debug(f"Memory initialized for session: {session_id}")
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加消息到短期记忆
        
        Args:
            role: 角色（user/assistant/system）
            content: 消息内容
            metadata: 元数据
        """
        message = Message(role, content, metadata=metadata)
        self.short_term_memory.append(message)
        
        # 如果超过最大消息数，移除最早的消息（保留系统消息）
        if len(self.short_term_memory) > self.max_short_term_messages:
            # 找到第一个非系统消息并移除
            for i, msg in enumerate(self.short_term_memory):
                if msg.role != "system":
                    self.short_term_memory.pop(i)
                    logger.debug(f"Removed oldest message to maintain memory limit")
                    break
        
        logger.debug(f"Added {role} message to memory (total: {len(self.short_term_memory)})")
    
    def get_conversation_history(
        self,
        include_system: bool = True,
        last_n: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            include_system: 是否包含系统消息
            last_n: 只返回最后n条消息
            
        Returns:
            消息列表，格式为 [{"role": "user", "content": "..."}]
        """
        messages = self.short_term_memory
        
        # 过滤系统消息
        if not include_system:
            messages = [msg for msg in messages if msg.role != "system"]
        
        # 只取最后n条
        if last_n is not None:
            messages = messages[-last_n:]
        
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def get_recent_context(self, max_messages: int = 10) -> str:
        """
        获取最近的对话上下文（格式化为文本）
        
        Args:
            max_messages: 最大消息数
            
        Returns:
            格式化的对话历史文本
        """
        recent_messages = self.short_term_memory[-max_messages:]
        
        context_parts = []
        for msg in recent_messages:
            if msg.role == "user":
                context_parts.append(f"用户: {msg.content}")
            elif msg.role == "assistant":
                context_parts.append(f"助手: {msg.content}")
        
        return "\n\n".join(context_parts)
    
    def add_to_long_term(
        self,
        category: str,
        key: str,
        value: Any
    ):
        """
        添加信息到长期记忆
        
        Args:
            category: 类别（user_preferences/task_context/important_facts等）
            key: 键
            value: 值
        """
        if category in self.long_term_memory:
            if isinstance(self.long_term_memory[category], dict):
                self.long_term_memory[category][key] = value
            elif isinstance(self.long_term_memory[category], list):
                self.long_term_memory[category].append({key: value})
            
            logger.debug(f"Added to long-term memory: {category}/{key}")
    
    def get_from_long_term(
        self,
        category: str,
        key: Optional[str] = None
    ) -> Any:
        """
        从长期记忆获取信息
        
        Args:
            category: 类别
            key: 键（可选）
            
        Returns:
            存储的值
        """
        if category not in self.long_term_memory:
            return None
        
        if key is None:
            return self.long_term_memory[category]
        
        if isinstance(self.long_term_memory[category], dict):
            return self.long_term_memory[category].get(key)
        
        return None
    
    def update_task_context(self, context: Dict[str, Any]):
        """
        更新任务上下文
        
        Args:
            context: 上下文信息
        """
        self.long_term_memory["task_context"].update(context)
        logger.debug(f"Updated task context: {list(context.keys())}")
    
    def record_generated_content(
        self,
        platform: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        记录生成的内容
        
        Args:
            platform: 平台名称
            content: 生成的内容
            metadata: 元数据
        """
        record = {
            "platform": platform,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.long_term_memory["generated_content"].append(record)
        logger.debug(f"Recorded generated content for {platform}")
    
    def get_generated_content_summary(self) -> str:
        """
        获取已生成内容的摘要
        
        Returns:
            摘要文本
        """
        contents = self.long_term_memory["generated_content"]
        if not contents:
            return "暂无生成记录"
        
        summary_parts = []
        for record in contents[-5:]:  # 只显示最近5条
            platform = record["platform"]
            timestamp = record["timestamp"]
            summary_parts.append(f"- {platform} ({timestamp})")
        
        return "\n".join(summary_parts)
    
    def clear_short_term(self):
        """清空短期记忆"""
        self.short_term_memory.clear()
        logger.debug("Short-term memory cleared")
    
    def clear_long_term(self):
        """清空长期记忆"""
        self.long_term_memory = {
            "user_preferences": {},
            "task_context": {},
            "important_facts": [],
            "generated_content": []
        }
        logger.debug("Long-term memory cleared")
    
    def clear_all(self):
        """清空所有记忆"""
        self.clear_short_term()
        self.clear_long_term()
        logger.debug("All memory cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "session_id": self.session_id,
            "short_term_messages": len(self.short_term_memory),
            "max_short_term_messages": self.max_short_term_messages,
            "long_term_categories": list(self.long_term_memory.keys()),
            "user_preferences_count": len(self.long_term_memory["user_preferences"]),
            "task_context_keys": list(self.long_term_memory["task_context"].keys()),
            "important_facts_count": len(self.long_term_memory["important_facts"]),
            "generated_content_count": len(self.long_term_memory["generated_content"])
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为字典（用于持久化）
        
        Returns:
            字典表示
        """
        return {
            "session_id": self.session_id,
            "max_short_term_messages": self.max_short_term_messages,
            "max_tokens": self.max_tokens,
            "short_term_memory": [msg.to_dict() for msg in self.short_term_memory],
            "long_term_memory": self.long_term_memory
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """
        从字典反序列化
        
        Args:
            data: 字典数据
            
        Returns:
            Memory实例
        """
        memory = cls(
            session_id=data["session_id"],
            max_short_term_messages=data.get("max_short_term_messages", 20),
            max_tokens=data.get("max_tokens", 4000)
        )
        
        # 恢复短期记忆
        memory.short_term_memory = [
            Message.from_dict(msg_data)
            for msg_data in data.get("short_term_memory", [])
        ]
        
        # 恢复长期记忆
        memory.long_term_memory = data.get("long_term_memory", {
            "user_preferences": {},
            "task_context": {},
            "important_facts": [],
            "generated_content": []
        })
        
        return memory
    
    def save_to_file(self, filepath: str):
        """
        保存到文件
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug(f"Memory saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional["Memory"]:
        """
        从文件加载
        
        Args:
            filepath: 文件路径
            
        Returns:
            Memory实例或None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Memory loaded from {filepath}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return None
