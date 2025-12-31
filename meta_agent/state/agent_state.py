"""
AgentState定义 - LangGraph状态管理
定义元agent和sub-agent之间共享的状态结构
"""
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from typing_extensions import NotRequired
import operator


class AgentInfo(TypedDict):
    """Agent信息"""
    agent_id: str
    agent_type: str  # template/llm/skill_based
    created_at: str
    status: str  # pending/running/completed/failed
    task: str
    config: Dict[str, Any]


class AgentState(TypedDict):
    """
    全局Agent状态
    使用TypedDict定义LangGraph的State结构
    """
    # ===== 全局共享状态 =====
    # 用户原始输入
    user_prompt: str
    
    # 任务目标（从prompt中提取）
    task_goal: str
    
    # 执行历史记录（使用operator.add支持追加）
    execution_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # 共享知识库（技能元数据、工具定义等）
    shared_knowledge: Dict[str, Any]
    
    # ===== Agent管理 =====
    # 活跃的agents信息
    active_agents: Dict[str, AgentInfo]
    
    # Agent执行结果
    agent_results: Dict[str, Any]
    
    # ===== 当前执行上下文 =====
    # 当前正在执行的agent ID
    current_agent_id: Optional[str]
    
    # 当前任务描述
    current_task: Optional[str]
    
    # ===== 技能和工具 =====
    # 已加载的技能列表
    loaded_skills: List[str]
    
    # 可用的工具列表
    available_tools: List[str]
    
    # ===== 元数据 =====
    # 会话ID
    session_id: str
    
    # 迭代次数
    iteration_count: int
    
    # 错误信息
    errors: Annotated[List[str], operator.add]
    
    # 最终结果
    final_result: NotRequired[str]

    # ===== Worker相关状态 =====
    # Worker指令列表（从Orchestrator解析出来的）
    worker_instructions: Annotated[List[Dict[str, Any]], operator.add]

    # Worker执行结果
    worker_results: Dict[str, Any]

    # 是否需要保存到本地
    save_to_local: bool

    # ===== Memory相关状态 =====
    # 对话历史（用于Memory模块）
    conversation_history: Annotated[List[Dict[str, str]], operator.add]

    # 任务上下文（长期记忆）
    task_context: Dict[str, Any]


class TaskStep(TypedDict):
    """单个任务步骤"""
    step_id: str
    worker_name: str  # Worker名称，如 @Douyin_Expert
    input_context: str  # 输入上下文
    status: str  # pending/running/completed/failed
    result: NotRequired[Dict[str, Any]]  # 执行结果

class PlanExecuteState(TypedDict):
    """
    Plan-Execute 模式的状态
    用于 Orchestrator-Worker 架构
    """
    # 用户原始输入
    user_prompt: str

    # 执行计划（由 Planner 生成的任务列表）
    plan: List[TaskStep]

    # 当前执行的步骤索引
    current_step_index: int

    # 所有步骤的执行结果
    step_results: Dict[str, Any]  # {step_id: result}

    # 是否需要重新规划
    need_replan: bool

    # 重新规划的原因
    replan_reason: NotRequired[str]

    # 会话ID
    session_id: str

    # 错误信息
    errors: Annotated[List[str], operator.add]

    # 最终输出
    final_output: NotRequired[str]

    # 共享知识（Skill Summary等）
    shared_knowledge: Dict[str, Any]

    # 是否需要保存到本地
    save_to_local: bool

    # 输出目录（如果保存到本地）
    output_directory: NotRequired[str]

    # 执行模式（chat/plan_execute）
    mode: NotRequired[str]

    # QA 质量检查报告
    qa_report: NotRequired[str]

def create_initial_state(user_prompt: str, session_id: str) -> AgentState:
    """
    创建初始状态

    Args:
        user_prompt: 用户输入的prompt
        session_id: 会话ID

    Returns:
        初始化的AgentState
    """
    return AgentState(
        user_prompt=user_prompt,
        task_goal="",
        execution_history=[],
        shared_knowledge={},
        active_agents={},
        agent_results={},
        current_agent_id=None,
        current_task=None,
        loaded_skills=[],
        available_tools=[],
        session_id=session_id,
        iteration_count=0,
        errors=[],
        worker_instructions=[],
        worker_results={},
        save_to_local=False,
        conversation_history=[],
        task_context={}
    )

def create_plan_execute_state(user_prompt: str, session_id: str) -> PlanExecuteState:
    """
    创建 Plan-Execute 模式的初始状态

    Args:
        user_prompt: 用户输入的prompt
        session_id: 会话ID

    Returns:
        初始化的PlanExecuteState
    """
    return PlanExecuteState(
        user_prompt=user_prompt,
        plan=[],
        current_step_index=0,
        step_results={},
        need_replan=False,
        session_id=session_id,
        errors=[],
        shared_knowledge={},
        save_to_local=False
    )
