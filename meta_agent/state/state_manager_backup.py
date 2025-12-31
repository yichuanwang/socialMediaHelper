"""
StateManager - 状态管理器
封装LangGraph的State操作，提供统一的状态管理接口
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from meta_agent.state.agent_state import AgentState, AgentInfo, create_initial_state


class StateManager:
    """
    状态管理器
    提供状态的创建、更新、查询等操作
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
    
    def update_task_goal(self, state: AgentState, task_goal: str) -> AgentState:
        """
        更新任务目标
        
        Args:
            state: 当前状态
            task_goal: 任务目标
            
        Returns:
            更新后的状态
        """
        state["task_goal"] = task_goal
        return state
    
    def add_agent(
        self,
        state: AgentState,
        agent_id: str,
        agent_type: str,
        task: str,
        config: Dict[str, Any]
    ) -> AgentState:
        """
        添加新的agent
        
        Args:
            state: 当前状态
            agent_id: agent ID
            agent_type: agent类型（template/llm/skill_based）
            task: 任务描述
            config: agent配置
            
        Returns:
            更新后的状态
        """
        agent_info: AgentInfo = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "task": task,
            "config": config
        }
        
        state["active_agents"][agent_id] = agent_info
        return state
    
    def update_agent_status(
        self,
        state: AgentState,
        agent_id: str,
        status: str
    ) -> AgentState:
        """
        更新agent状态
        
        Args:
            state: 当前状态
            agent_id: agent ID
            status: 新状态（pending/running/completed/failed）
            
        Returns:
            更新后的状态
        """
        if agent_id in state["active_agents"]:
            state["active_agents"][agent_id]["status"] = status
        return state
    
    def set_current_agent(
        self,
        state: AgentState,
        agent_id: Optional[str],
        task: Optional[str] = None
    ) -> AgentState:
        """
        设置当前执行的agent
        
        Args:
            state: 当前状态
            agent_id: agent ID
            task: 当前任务
            
        Returns:
            更新后的状态
        """
        state["current_agent_id"] = agent_id
        if task:
            state["current_task"] = task
        return state
    
    def add_agent_result(
        self,
        state: AgentState,
        agent_id: str,
        result: Any
    ) -> AgentState:
        """
        添加agent执行结果
        
        Args:
            state: 当前状态
            agent_id: agent ID
            result: 执行结果
            
        Returns:
            更新后的状态
        """
        state["agent_results"][agent_id] = result
        return state
    
    def add_execution_record(
        self,
        state: AgentState,
        record: Dict[str, Any]
    ) -> AgentState:
        """
        添加执行记录
        
        Args:
            state: 当前状态
            record: 执行记录
            
        Returns:
            更新后的状态
        """
        # 添加时间戳
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()
            
        state["execution_history"].append(record)
        return state
    
    def load_skill(self, state: AgentState, skill_name: str) -> AgentState:
        """
        加载技能
        
        Args:
            state: 当前状态
            skill_name: 技能名称
            
        Returns:
            更新后的状态
        """
        if skill_name not in state["loaded_skills"]:
            state["loaded_skills"].append(skill_name)
        return state
    
    def register_tool(self, state: AgentState, tool_name: str) -> AgentState:
        """
        注册工具
        
        Args:
            state: 当前状态
            tool_name: 工具名称
            
        Returns:
            更新后的状态
        """
        if tool_name not in state["available_tools"]:
            state["available_tools"].append(tool_name)
        return state
    
    def update_shared_knowledge(
        self,
        state: AgentState,
        key: str,
        value: Any
    ) -> AgentState:
        """
        更新共享知识
        
        Args:
            state: 当前状态
            key: 知识键
            value: 知识值
            
        Returns:
            更新后的状态
        """
        state["shared_knowledge"][key] = value
        return state
    
    def increment_iteration(self, state: AgentState) -> AgentState:
        """
        增加迭代计数
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        state["iteration_count"] += 1
        return state
    
    def add_error(self, state: AgentState, error: str) -> AgentState:
        """
        添加错误信息
        
        Args:
            state: 当前状态
            error: 错误信息
            
        Returns:
            更新后的状态
        """
        state["errors"].append(error)
        return state
    
    def set_final_result(self, state: AgentState, result: str) -> AgentState:
        """
        设置最终结果
        
        Args:
            state: 当前状态
            result: 最终结果
            
        Returns:
            更新后的状态
        """
        state["final_result"] = result
        return state
    
    def get_agent_info(
        self,
        state: AgentState,
        agent_id: str
    ) -> Optional[AgentInfo]:
        """
        获取agent信息
        
        Args:
            state: 当前状态
            agent_id: agent ID
            
        Returns:
            agent信息或None
        """
        return state["active_agents"].get(agent_id)
    
    def get_agent_result(
        self,
        state: AgentState,
        agent_id: str
    ) -> Optional[Any]:
        """
        获取agent结果
        
        Args:
            state: 当前状态
            agent_id: agent ID
            
        Returns:
            agent结果或None
        """
        return state["agent_results"].get(agent_id)
    
    def get_all_agent_results(self, state: AgentState) -> Dict[str, Any]:
        """
        获取所有agent结果
        
        Args:
            state: 当前状态
            
        Returns:
            所有agent结果
        """
        return state["agent_results"]
    
    def get_execution_history(
        self,
        state: AgentState,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取执行历史
        
        Args:
            state: 当前状态
            limit: 限制数量
            
        Returns:
            执行历史列表
        """
        history = state["execution_history"]
        if limit:
            return history[-limit:]
        return history
    
    def is_agent_completed(self, state: AgentState, agent_id: str) -> bool:
        """
        检查agent是否完成
        
        Args:
            state: 当前状态
            agent_id: agent ID
            
        Returns:
            是否完成
        """
        agent_info = self.get_agent_info(state, agent_id)
        if agent_info:
            return agent_info["status"] == "completed"
        return False
    
    def get_pending_agents(self, state: AgentState) -> List[str]:
        """
        获取待执行的agent列表
        
        Args:
            state: 当前状态
            
        Returns:
            待执行agent ID列表
        """
        return [
            agent_id for agent_id, info in state["active_agents"].items()
            if info["status"] == "pending"
        ]
    
    def get_running_agents(self, state: AgentState) -> List[str]:
        """
        获取正在运行的agent列表
        
        Args:
            state: 当前状态
            
        Returns:
            运行中agent ID列表
        """
        return [
            agent_id for agent_id, info in state["active_agents"].items()
            if info["status"] == "running"
        ]
    
    def _generate_session_id(self) -> str:
        """
        生成会话ID
        
        Returns:
            会话ID
        """
        return f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _generate_agent_id(self, agent_type: str) -> str:
        """
        生成agent ID
        
        Args:
            agent_type: agent类型
            
        Returns:
            agent ID
        """
        return f"{agent_type}_{uuid.uuid4().hex[:8]}"
