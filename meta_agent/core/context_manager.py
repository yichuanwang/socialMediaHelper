"""
ContextManager - 上下文管理器
负责管理meta-agent和sub-agent之间的上下文共享
实现渐进式披露和上下文压缩
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class ContextManager:
    """
    上下文管理器
    实现分层上下文管理和渐进式披露
    """
    
    def __init__(self, max_history_length: int = 50):
        """
        初始化上下文管理器
        
        Args:
            max_history_length: 最大历史记录长度
        """
        self.max_history_length = max_history_length
        
    def create_agent_context(
        self,
        state: Dict[str, Any],
        agent_id: str,
        task: str,
        relevant_skills: Optional[List[str]] = None,
        relevant_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        为sub-agent创建上下文
        实现选择性传递，只传递相关信息
        
        Args:
            state: 全局状态
            agent_id: agent ID
            task: 任务描述
            relevant_skills: 相关技能列表
            relevant_tools: 相关工具列表
            
        Returns:
            agent上下文字典
        """
        # 基础上下文
        context = {
            "agent_id": agent_id,
            "task": task,
            "user_prompt": state.get("user_prompt", ""),
            "task_goal": state.get("task_goal", ""),
            "session_id": state.get("session_id", ""),
        }
        
        # 添加相关的执行历史（压缩后的）
        history = state.get("execution_history", [])
        if history:
            context["relevant_history"] = self._compress_history(
                history,
                max_items=10
            )
        
        # 添加相关技能信息（仅元数据）
        if relevant_skills:
            context["available_skills"] = relevant_skills
            
        # 添加相关工具信息
        if relevant_tools:
            context["available_tools"] = relevant_tools
            
        # 添加其他agent的结果（如果相关）
        agent_results = state.get("agent_results", {})
        if agent_results:
            context["other_agent_results"] = {
                aid: result for aid, result in agent_results.items()
                if aid != agent_id
            }
            
        return context
    
    def update_global_context(
        self,
        state: Dict[str, Any],
        agent_id: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        更新全局上下文
        将sub-agent的结果合并到全局状态
        
        Args:
            state: 全局状态
            agent_id: agent ID
            result: agent执行结果
            metadata: 额外的元数据
            
        Returns:
            更新后的状态
        """
        # 更新agent结果
        if "agent_results" not in state:
            state["agent_results"] = {}
        state["agent_results"][agent_id] = result
        
        # 添加到执行历史
        history_entry = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "result": self._summarize_result(result),
            "metadata": metadata or {}
        }
        
        if "execution_history" not in state:
            state["execution_history"] = []
        state["execution_history"].append(history_entry)
        
        # 如果历史记录过长，进行压缩
        if len(state["execution_history"]) > self.max_history_length:
            state["execution_history"] = self._compress_history(
                state["execution_history"],
                max_items=self.max_history_length // 2
            )
            
        # 更新agent状态
        if "active_agents" in state and agent_id in state["active_agents"]:
            state["active_agents"][agent_id]["status"] = "completed"
            
        return state
    
    def _compress_history(
        self,
        history: List[Dict[str, Any]],
        max_items: int
    ) -> List[Dict[str, Any]]:
        """
        压缩历史记录
        保留最近的记录和重要的记录
        
        Args:
            history: 历史记录列表
            max_items: 最大保留数量
            
        Returns:
            压缩后的历史记录
        """
        if len(history) <= max_items:
            return history
            
        # 保留最近的记录
        recent_items = history[-max_items:]
        
        # 可以在这里添加更复杂的压缩逻辑
        # 例如：保留重要的里程碑记录
        
        return recent_items
    
    def _summarize_result(self, result: Any) -> str:
        """
        总结结果
        将复杂结果转换为简短摘要
        
        Args:
            result: 原始结果
            
        Returns:
            结果摘要
        """
        if isinstance(result, str):
            # 如果结果太长，截断
            if len(result) > 500:
                return result[:500] + "..."
            return result
        elif isinstance(result, dict):
            # 提取关键信息
            summary = {
                k: v for k, v in result.items()
                if k in ["status", "summary", "key_points", "error"]
            }
            return json.dumps(summary, ensure_ascii=False)
        else:
            return str(result)[:500]
    
    def get_shared_knowledge(
        self,
        state: Dict[str, Any],
        knowledge_type: str
    ) -> Any:
        """
        获取共享知识
        
        Args:
            state: 全局状态
            knowledge_type: 知识类型（如：skills, tools）
            
        Returns:
            知识内容
        """
        shared_knowledge = state.get("shared_knowledge", {})
        return shared_knowledge.get(knowledge_type)
    
    def update_shared_knowledge(
        self,
        state: Dict[str, Any],
        knowledge_type: str,
        knowledge: Any
    ) -> Dict[str, Any]:
        """
        更新共享知识
        
        Args:
            state: 全局状态
            knowledge_type: 知识类型
            knowledge: 知识内容
            
        Returns:
            更新后的状态
        """
        if "shared_knowledge" not in state:
            state["shared_knowledge"] = {}
        state["shared_knowledge"][knowledge_type] = knowledge
        return state
    
    def extract_relevant_context(
        self,
        state: Dict[str, Any],
        query: str,
        context_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        提取相关上下文
        根据查询提取最相关的上下文信息
        
        Args:
            state: 全局状态
            query: 查询字符串
            context_types: 需要的上下文类型列表
            
        Returns:
            相关上下文
        """
        relevant_context = {}
        
        if not context_types:
            context_types = ["history", "skills", "tools", "results"]
            
        # 提取历史记录
        if "history" in context_types:
            history = state.get("execution_history", [])
            # 简单实现：返回最近的几条
            relevant_context["history"] = history[-5:] if history else []
            
        # 提取技能信息
        if "skills" in context_types:
            relevant_context["skills"] = state.get("loaded_skills", [])
            
        # 提取工具信息
        if "tools" in context_types:
            relevant_context["tools"] = state.get("available_tools", [])
            
        # 提取agent结果
        if "results" in context_types:
            relevant_context["results"] = state.get("agent_results", {})
            
        return relevant_context
