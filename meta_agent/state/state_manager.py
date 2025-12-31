"""
StateManager - 状态管理器（重构版）
封装LangGraph的State操作，提供统一的状态管理接口
"""
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid

from meta_agent.state.agent_state import AgentState, AgentInfo, create_initial_state

class StateManager:
    """
    状态管理器（重构版）
    提供统一的状态更新接口，减少方法数量
    """
    
    def __init__(self):
        """初始化状态管理器"""
        pass
    
    def create_state(self, user_prompt: str) -> AgentState:
        """
        创建新的状态
        
        Args:
            user_prompt: 用户输入
            
        Returns:
            初始化的AgentState
        """
        session_id = self._generate_session_id()
        return create_initial_state(user_prompt, session_id)
    
    def update_state(
        self,
        state: AgentState,
        **updates: Any
    ) -> AgentState:
        """
        统一的状态更新方法
        
        支持的更新操作：
        - task_goal: 更新任务目标
        - current_agent_id: 设置当前agent
        - current_task: 设置当前任务
        - final_result: 设置最终结果
        - iteration_count: 增加迭代计数（传入 increment=True）
        - 其他任何状态字段的直接更新
        
        Args:
            state: 当前状态
            **updates: 要更新的字段和值
            
        Returns:
            更新后的状态
            
        Examples:
            # 更新任务目标
            state = manager.update_state(state, task_goal="新目标")
            
            # 设置当前agent和任务
            state = manager.update_state(
                state,
                current_agent_id="agent_123",
                current_task="执行任务"
            )
            
            # 增加迭代计数
            state = manager.update_state(state, iteration_count_increment=True)
            
            # 设置最终结果
            state = manager.update_state(state, final_result="完成")
        """
        for key, value in updates.items():
            # 特殊处理：迭代计数增加
            if key == "iteration_count_increment" and value:
                state["iteration_count"] += 1
            # 普通字段更新
            elif key in state:
                state[key] = value
        
        return state
    
    def update_agent(
        self,
        state: AgentState,
        agent_id: str,
        **updates: Any
    ) -> AgentState:
        """
        统一的agent更新方法
        
        支持的操作：
        - 添加新agent（传入 agent_type, task, config）
        - 更新agent状态（传入 status）
        - 更新agent的任何字段
        
        Args:
            state: 当前状态
            agent_id: agent ID
            **updates: 要更新的字段
            
        Returns:
            更新后的状态
            
        Examples:
            # 添加新agent
            state = manager.update_agent(
                state,
                "agent_123",
                agent_type="skill_based",
                task="执行任务",
                config={}
            )
            
            # 更新agent状态
            state = manager.update_agent(state, "agent_123", status="running")
            
            # 同时更新多个字段
            state = manager.update_agent(
                state,
                "agent_123",
                status="completed",
                result="成功"
            )
        """
        # 如果agent不存在且提供了创建所需的字段，则创建新agent
        if agent_id not in state["active_agents"]:
            if "agent_type" in updates and "task" in updates:
                agent_info: AgentInfo = {
                    "agent_id": agent_id,
                    "agent_type": updates.pop("agent_type"),
                    "created_at": datetime.now().isoformat(),
                    "status": updates.pop("status", "pending"),
                    "task": updates.pop("task"),
                    "config": updates.pop("config", {})
                }
                # 添加其他额外字段
                agent_info.update(updates)
                state["active_agents"][agent_id] = agent_info
            else:
                raise ValueError(f"Agent {agent_id} does not exist. To create, provide agent_type and task.")
        else:
            # 更新现有agent
            for key, value in updates.items():
                state["active_agents"][agent_id][key] = value
        
        return state
    
    def add_to_list(
        self,
        state: AgentState,
        list_name: str,
        item: Any,
        unique: bool = False
    ) -> AgentState:
        """
        向列表类型的状态字段添加项
        
        Args:
            state: 当前状态
            list_name: 列表字段名（如 loaded_skills, available_tools, errors等）
            item: 要添加的项
            unique: 是否确保唯一性（不重复添加）
            
        Returns:
            更新后的状态
            
        Examples:
            # 加载技能
            state = manager.add_to_list(state, "loaded_skills", "content-generate", unique=True)
            
            # 注册工具
            state = manager.add_to_list(state, "available_tools", "search_tool", unique=True)
            
            # 添加错误
            state = manager.add_to_list(state, "errors", "执行失败")
            
            # 添加执行记录
            state = manager.add_to_list(state, "execution_history", {"action": "test"})
        """
        if list_name not in state:
            raise ValueError(f"List field '{list_name}' does not exist in state")
        
        if not isinstance(state[list_name], list):
            raise ValueError(f"Field '{list_name}' is not a list")
        
        # 添加时间戳（如果是执行记录）
        if list_name == "execution_history" and isinstance(item, dict):
            if "timestamp" not in item:
                item["timestamp"] = datetime.now().isoformat()
        
        # 检查唯一性
        if unique and item in state[list_name]:
            return state
        
        state[list_name].append(item)
        return state
    
    def update_dict(
        self,
        state: AgentState,
        dict_name: str,
        key: str,
        value: Any
    ) -> AgentState:
        """
        更新字典类型的状态字段
        
        Args:
            state: 当前状态
            dict_name: 字典字段名（如 agent_results, shared_knowledge等）
            key: 字典键
            value: 字典值
            
        Returns:
            更新后的状态
            
        Examples:
            # 添加agent结果
            state = manager.update_dict(state, "agent_results", "agent_123", result_data)
            
            # 更新共享知识
            state = manager.update_dict(state, "shared_knowledge", "context", context_data)
        """
        if dict_name not in state:
            raise ValueError(f"Dict field '{dict_name}' does not exist in state")
        
        if not isinstance(state[dict_name], dict):
            raise ValueError(f"Field '{dict_name}' is not a dict")
        
        state[dict_name][key] = value
        return state
    
    # ===== 向后兼容的便捷方法 =====
    
    def update_task_goal(self, state: AgentState, task_goal: str) -> AgentState:
        """更新任务目标（兼容方法）"""
        return self.update_state(state, task_goal=task_goal)
    
    def add_agent(
        self,
        state: AgentState,
        agent_id: str,
        agent_type: str,
        task: str,
        config: Dict[str, Any]
    ) -> AgentState:
        """添加新agent（兼容方法）"""
        return self.update_agent(
            state,
            agent_id,
            agent_type=agent_type,
            task=task,
            config=config
        )
    
    def update_agent_status(
        self,
        state: AgentState,
        agent_id: str,
        status: str
    ) -> AgentState:
        """更新agent状态（兼容方法）"""
        return self.update_agent(state, agent_id, status=status)
    
    def set_current_agent(
        self,
        state: AgentState,
        agent_id: Optional[str],
        task: Optional[str] = None
    ) -> AgentState:
        """设置当前agent（兼容方法）"""
        updates = {"current_agent_id": agent_id}
        if task:
            updates["current_task"] = task
        return self.update_state(state, **updates)
    
    def add_agent_result(
        self,
        state: AgentState,
        agent_id: str,
        result: Any
    ) -> AgentState:
        """添加agent结果（兼容方法）"""
        return self.update_dict(state, "agent_results", agent_id, result)
    
    def add_execution_record(
        self,
        state: AgentState,
        record: Dict[str, Any]
    ) -> AgentState:
        """添加执行记录（兼容方法）"""
        return self.add_to_list(state, "execution_history", record)
    
    def load_skill(self, state: AgentState, skill_name: str) -> AgentState:
        """加载技能（兼容方法）"""
        return self.add_to_list(state, "loaded_skills", skill_name, unique=True)
    
    def register_tool(self, state: AgentState, tool_name: str) -> AgentState:
        """注册工具（兼容方法）"""
        return self.add_to_list(state, "available_tools", tool_name, unique=True)
    
    def update_shared_knowledge(
        self,
        state: AgentState,
        key: str,
        value: Any
    ) -> AgentState:
        """更新共享知识（兼容方法）"""
        return self.update_dict(state, "shared_knowledge", key, value)
    
    def increment_iteration(self, state: AgentState) -> AgentState:
        """增加迭代计数（兼容方法）"""
        return self.update_state(state, iteration_count_increment=True)
    
    def add_error(self, state: AgentState, error: str) -> AgentState:
        """添加错误（兼容方法）"""
        return self.add_to_list(state, "errors", error)
    
    def set_final_result(self, state: AgentState, result: str) -> AgentState:
        """设置最终结果（兼容方法）"""
        return self.update_state(state, final_result=result)
    
    # ===== 查询方法（保持不变）=====
    
    def get_agent_info(
        self,
        state: AgentState,
        agent_id: str
    ) -> Optional[AgentInfo]:
        """获取agent信息"""
        return state["active_agents"].get(agent_id)
    
    def get_agent_result(
        self,
        state: AgentState,
        agent_id: str
    ) -> Optional[Any]:
        """获取agent结果"""
        return state["agent_results"].get(agent_id)
    
    def get_all_agent_results(self, state: AgentState) -> Dict[str, Any]:
        """获取所有agent结果"""
        return state["agent_results"]
    
    def get_execution_history(
        self,
        state: AgentState,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """获取执行历史"""
        history = state["execution_history"]
        if limit:
            return history[-limit:]
        return history
    
    def is_agent_completed(self, state: AgentState, agent_id: str) -> bool:
        """检查agent是否完成"""
        agent_info = self.get_agent_info(state, agent_id)
        if agent_info:
            return agent_info["status"] == "completed"
        return False
    
    def get_pending_agents(self, state: AgentState) -> List[str]:
        """获取待执行的agent列表"""
        return [
            agent_id for agent_id, info in state["active_agents"].items()
            if info["status"] == "pending"
        ]
    
    def get_running_agents(self, state: AgentState) -> List[str]:
        """获取正在运行的agent列表"""
        return [
            agent_id for agent_id, info in state["active_agents"].items()
            if info["status"] == "running"
        ]
    
    # ===== 私有方法 =====
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _generate_agent_id(self, agent_type: str) -> str:
        """生成agent ID"""
        return f"{agent_type}_{uuid.uuid4().hex[:8]}"
