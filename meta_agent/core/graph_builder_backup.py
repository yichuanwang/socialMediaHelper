"""
GraphBuilder - 使用LangGraph v1.0构建Agent执行图
"""
from typing import Dict, Any, Optional, Literal, List
import logging
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from ..state.agent_state import AgentState, PlanExecuteState, TaskStep
from ..core.skill_agent import SkillAgent
from ..utils.prompt_parser import PromptParser


logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    图构建器
    使用LangGraph构建Agent的执行流程图
    """
    
    def __init__(self, skill_agent: SkillAgent):
        """
        初始化图构建器
        
        Args:
            skill_agent: SkillAgent实例
        """
        self.skill_agent = skill_agent
        self.prompt_parser = PromptParser()
        self.graph: Optional[StateGraph] = None
        
        logger.debug("GraphBuilder initialized")
    
    def build_graph(self, mode: str = "content_generate") -> StateGraph:
        """
        构建执行图

        Args:
            mode: 图模式，"plan_execute"、"content_generate" 或 "general"

        Returns:
            构建好的StateGraph
        """
        if mode == "plan_execute":
            return self._build_plan_execute_graph()
        elif mode == "content_generate":
            return self._build_content_generate_graph()
        else:
            return self._build_general_graph()

    def _build_plan_execute_graph(self) -> StateGraph:
        """
        构建 Plan-Execute 模式的图

        Returns:
            编译后的图
        """
        logger.debug("Building Plan-Execute graph")

        # 导入状态类型
        from meta_agent.state.agent_state import PlanExecuteState

        # 创建图
        graph = StateGraph(PlanExecuteState)

        # 添加节点
        graph.add_node("router", self._router_node)  # Layer 1: 路由节点
        graph.add_node("chat_response", self._chat_response_node)  # 直接对话节点
        graph.add_node("tool_call", self._tool_call_node)  # 工具调用节点
        graph.add_node("planner", self._planner_node)  # Layer 2: 规划节点
        graph.add_node("worker", self._worker_node)
        graph.add_node("replan", self._replan_node)
        graph.add_node("qa", self._qa_node)  # QA 质量检查节点

        # 设置入口点
        graph.set_entry_point("router")

        # 添加条件边：router 决定下一步
        graph.add_conditional_edges(
            "router",
            self._should_continue_after_router,
            {
                "planner": "planner",  # 需要调用 skill
                "tool_call": "tool_call",  # 需要调用工具
                "chat_response": "chat_response"  # 直接对话回复
            }
        )

        # chat_response 和 tool_call 节点直接结束
        graph.add_edge("chat_response", END)
        graph.add_edge("tool_call", END)

        # 添加边
        graph.add_edge("planner", "worker")
        graph.add_edge("worker", "replan")

        # 添加条件边：replan 后决定下一步
        graph.add_conditional_edges(
            "replan",
            self._should_continue_after_replan,
            {
                "worker": "worker",              # 继续执行下一个任务
                "planner": "planner",            # 重新规划
                "qa": "qa",                      # 质量检查
                "end": END                       # 直接结束
            }
        )

        # QA 节点后直接结束
        graph.add_edge("qa", END)

        # 编译图
        compiled_graph = graph.compile()
        logger.debug("Plan-Execute graph compiled successfully")

        return compiled_graph

    def _build_content_generate_graph(self) -> StateGraph:
        """
        构建Content-Generate专用图（Orchestrator-Worker架构）

        Returns:
            构建好的StateGraph
        """
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("intercept_instructions", self._intercept_instructions_node)
        workflow.add_node("execute_workers", self._execute_workers_node)
        workflow.add_node("save_to_local", self._save_to_local_node)

        # 设置入口点
        workflow.set_entry_point("orchestrator")

        # 添加边
        workflow.add_edge("orchestrator", "intercept_instructions")
        workflow.add_conditional_edges(
            "intercept_instructions",
            self._should_execute_workers,
            {
                "execute_workers": "execute_workers",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "execute_workers",
            self._should_save_to_local,
            {
                "save_to_local": "save_to_local",
                "end": END
            }
        )
        workflow.add_edge("save_to_local", END)

        # 编译图
        self.graph = workflow.compile()
        logger.debug("Content-Generate graph built successfully")
        return self.graph

    def _build_general_graph(self) -> StateGraph:
        """
        构建通用执行图（旧的流程）

        Returns:
            构建好的StateGraph
        """
        # 创建StateGraph
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("parse_prompt", self._parse_prompt_node)
        workflow.add_node("select_skills", self._select_skills_node)
        workflow.add_node("load_skill", self._load_skill_node)
        workflow.add_node("execute_skill", self._execute_skill_node)
        workflow.add_node("aggregate_results", self._aggregate_results_node)

        # 设置入口点
        workflow.set_entry_point("parse_prompt")

        # 添加边
        workflow.add_edge("parse_prompt", "select_skills")
        workflow.add_conditional_edges(
            "select_skills",
            self._should_continue_after_selection,
            {
                "load_skill": "load_skill",
                "end": END
            }
        )
        workflow.add_edge("load_skill", "execute_skill")
        workflow.add_conditional_edges(
            "execute_skill",
            self._should_continue_after_execution,
            {
                "load_skill": "load_skill",
                "aggregate": "aggregate_results"
            }
        )
        workflow.add_edge("aggregate_results", END)

        # 编译图
        self.graph = workflow.compile()

        logger.debug("General graph built successfully")
        return self.graph
    
    async def _parse_prompt_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        解析Prompt节点
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新
        """
        logger.debug("Executing parse_prompt node")
        
        try:
            # 解析prompt
            parsed_prompt = self.prompt_parser.parse(state["user_prompt"])
            
            # 更新状态
            return {
                "task_goal": parsed_prompt.task_goal,
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "parsed_prompt": {
                        "task_type": parsed_prompt.task_type,
                        "complexity": parsed_prompt.complexity,
                        "priority": parsed_prompt.priority,
                        "required_skills": parsed_prompt.required_skills,
                        "entities": parsed_prompt.entities,
                        "constraints": parsed_prompt.constraints
                    }
                },
                "iteration_count": state.get("iteration_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Error in parse_prompt node: {e}")
            return {
                "errors": [f"Parse error: {str(e)}"]
            }
    
    async def _select_skills_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        选择技能节点
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新
        """
        logger.debug("Executing select_skills node")
        
        try:
            # 从共享知识中获取解析结果
            parsed_data = state.get("shared_knowledge", {}).get("parsed_prompt", {})
            required_skills = parsed_data.get("required_skills", [])
            
            # 如果没有推荐的技能，使用技能注册表搜索
            if not required_skills:
                task_goal = state.get("task_goal", "")
                search_results = self.skill_agent.skill_registry.search_skills(task_goal)
                if search_results:
                    required_skills = [search_results[0].name]
            
            # 更新状态
            return {
                "loaded_skills": required_skills,
                "current_task": f"Execute skills: {', '.join(required_skills)}",
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "selected_skills": required_skills,
                    "current_skill_index": 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in select_skills node: {e}")
            return {
                "errors": [f"Skill selection error: {str(e)}"]
            }
    
    async def _load_skill_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        加载技能节点
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新
        """
        logger.debug("Executing load_skill node")
        
        try:
            # 获取当前要加载的技能
            shared_knowledge = state.get("shared_knowledge", {})
            selected_skills = shared_knowledge.get("selected_skills", [])
            current_index = shared_knowledge.get("current_skill_index", 0)
            
            if current_index >= len(selected_skills):
                return {"errors": ["No more skills to load"]}
            
            skill_name = selected_skills[current_index]
            
            # 加载技能
            success = await self.skill_agent.load_skill(skill_name)
            
            if not success:
                return {
                    "errors": [f"Failed to load skill: {skill_name}"]
                }
            
            return {
                "shared_knowledge": {
                    **shared_knowledge,
                    "current_skill": skill_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error in load_skill node: {e}")
            return {
                "errors": [f"Skill loading error: {str(e)}"]
            }
    
    async def _execute_skill_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        执行技能节点
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新
        """
        logger.debug("Executing execute_skill node")
        
        try:
            # 获取当前技能
            shared_knowledge = state.get("shared_knowledge", {})
            current_skill = shared_knowledge.get("current_skill")
            
            if not current_skill:
                return {"errors": ["No current skill to execute"]}
            
            # 准备执行上下文
            parsed_data = shared_knowledge.get("parsed_prompt", {})
            context = {
                "task_goal": state.get("task_goal", ""),
                "task_type": parsed_data.get("task_type", ""),
                "entities": parsed_data.get("entities", {}),
                "constraints": parsed_data.get("constraints", []),
                "original_prompt": state.get("user_prompt", "")
            }
            
            # 执行技能
            result = await self.skill_agent.skill_executor.execute_skill(
                current_skill,
                context,
                state
            )
            
            # 更新执行历史
            execution_history = state.get("execution_history", [])
            execution_history.append({
                "skill": current_skill,
                "success": result.success,
                "result": result.result if result.success else None,
                "error": result.error if not result.success else None
            })
            
            # 更新agent结果
            agent_results = state.get("agent_results", {})
            agent_results[current_skill] = result.to_dict()
            
            # 增加技能索引
            current_index = shared_knowledge.get("current_skill_index", 0)
            
            return {
                "execution_history": execution_history,
                "agent_results": agent_results,
                "shared_knowledge": {
                    **shared_knowledge,
                    "current_skill_index": current_index + 1,
                    "last_execution_success": result.success
                }
            }
            
        except Exception as e:
            logger.error(f"Error in execute_skill node: {e}")
            return {
                "errors": [f"Skill execution error: {str(e)}"]
            }
    
    async def _aggregate_results_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        聚合结果节点
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新
        """
        logger.debug("Executing aggregate_results node")
        
        try:
            # 收集所有执行结果
            agent_results = state.get("agent_results", {})
            execution_history = state.get("execution_history", [])
            
            # 检查是否所有技能都成功执行
            all_success = all(
                result.get("success", False)
                for result in agent_results.values()
            )
            
            # 生成最终结果摘要
            summary_parts = []
            if all_success:
                summary_parts.append(f"Successfully completed task: {state.get('task_goal', '')}")
                summary_parts.append(f"Executed {len(agent_results)} skill(s)")
            else:
                failed_skills = [
                    name for name, result in agent_results.items()
                    if not result.get("success", False)
                ]
                summary_parts.append(f"Task partially completed")
                summary_parts.append(f"Failed skills: {', '.join(failed_skills)}")
            
            final_result = " | ".join(summary_parts)
            
            return {
                "final_result": final_result,
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "execution_complete": True,
                    "all_success": all_success
                }
            }
            
        except Exception as e:
            logger.error(f"Error in aggregate_results node: {e}")
            return {
                "errors": [f"Result aggregation error: {str(e)}"],
                "final_result": "Error occurred during result aggregation"
            }
    
    def _should_continue_after_selection(
        self,
        state: AgentState
    ) -> Literal["load_skill", "end"]:
        """
        判断选择技能后是否继续
        
        Args:
            state: 当前状态
            
        Returns:
            下一个节点名称
        """
        selected_skills = state.get("shared_knowledge", {}).get("selected_skills", [])
        
        if not selected_skills:
            logger.warning("No skills selected, ending execution")
            return "end"
        
        return "load_skill"
    
    def _should_continue_after_execution(
        self,
        state: AgentState
    ) -> Literal["load_skill", "aggregate"]:
        """
        判断执行技能后是否继续
        
        Args:
            state: 当前状态
            
        Returns:
            下一个节点名称
        """
        shared_knowledge = state.get("shared_knowledge", {})
        selected_skills = shared_knowledge.get("selected_skills", [])
        current_index = shared_knowledge.get("current_skill_index", 0)
        
        # 如果还有更多技能要执行
        if current_index < len(selected_skills):
            logger.debug(f"Continuing to next skill ({current_index + 1}/{len(selected_skills)})")
            return "load_skill"
        
        # 所有技能都执行完毕
        logger.debug("All skills executed, aggregating results")
        return "aggregate"
    
    def get_graph_visualization(self) -> str:
        """
        获取图的可视化表示
        
        Returns:
            图的文本表示
        """
        if not self.graph:
            return "Graph not built yet"
        
        return """
        Agent Execution Graph:
        
        START
          ↓
        [parse_prompt] - 解析用户prompt
          ↓
        [select_skills] - 选择所需技能
          ↓
        (decision: has skills?)
          ├─ Yes → [load_skill] - 加载技能
          │           ↓
          │         [execute_skill] - 执行技能
          │           ↓
          │         (decision: more skills?)
          │           ├─ Yes → [load_skill] (循环)
          │           └─ No → [aggregate_results] - 聚合结果
          │                     ↓
          │                   END
          └─ No → END
        """
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        获取图的统计信息
        
        Returns:
            统计信息字典
        """
        if not self.graph:
            return {"status": "not_built"}
        
        return {
            "status": "built",
            "nodes": [
                "parse_prompt",
                "select_skills",
                "load_skill",
                "execute_skill",
                "aggregate_results"
            ],
            "node_count": 5,
            "has_conditional_edges": True,
            "has_loops": True,
            "entry_point": "parse_prompt"
        }

    # ===== Content-Generate专用节点 =====

    async def _orchestrator_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Orchestrator节点 - 分析需求并输出Worker指令

        这个节点加载content-generate技能的Summary层，
        作为Orchestrator来分析用户需求并决定调用哪些Worker

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含Orchestrator的回复（含Worker指令）
        """
        logger.debug("Executing orchestrator node")

        try:
            # 加载content-generate技能的Frontmatter和Summary
            skill_name = "content-generate"
            skill_loader = self.skill_agent.skill_loader

            # 使用新方法加载Frontmatter和Summary
            skill_data = skill_loader.load_skill_with_metadata(skill_name)
            if not skill_data:
                logger.debug(f"Failed to load skill data: {skill_name}, using fallback")
                # 如果已经有orchestrator_response，直接使用
                existing_response = state.get("shared_knowledge", {}).get("orchestrator_response")
                if existing_response:
                    logger.debug("Using existing orchestrator_response from state")
                    return {
                        "shared_knowledge": {
                            **state.get("shared_knowledge", {}),
                            "orchestrator_response": existing_response
                        }
                    }
                # 否则返回错误
                return {
                    "errors": [f"Failed to load skill data: {skill_name}"]
                }

            # 提取metadata和summary
            metadata = skill_data.get("metadata", {})
            skill_summary = skill_data.get("summary", "")

            # 格式化Frontmatter为可读文本
            metadata_text = self._format_skill_metadata(metadata)

            # 加载主Agent的System Prompt
            from meta_agent.config.agent_prompts import get_agent_prompt
            agent_prompt = get_agent_prompt()

            # 组合主Agent Prompt、Frontmatter和技能Summary
            combined_prompt = f"""{agent_prompt}

---

## 当前技能信息

{metadata_text}

## 技能详情

{skill_summary}

---

请根据上述角色定位和技能说明，处理用户的请求。"""

            # 调用LLM生成Orchestrator回复（使用拦截机制）
            from meta_agent.core.llm_manager import LLMManager
            from meta_agent.core.skill_interceptor import SkillInterceptor
            from pathlib import Path

            llm = LLMManager()

            # 初始化拦截器
            skills_dir = Path("meta_agent/skills")
            interceptor = SkillInterceptor(skills_dir)

            user_prompt = state.get("user_prompt", "")

            # 获取对话历史（从 Memory）- 包含 ReAct 循环中的上下文
            memory = self.skill_agent.memory
            conversation_history = memory.get_conversation_history(
                include_system=False,
                last_n=10  # 最近10条消息
            )

            # 如果没有历史记录，使用原始 user_prompt
            if not conversation_history:
                conversation_history = [{"role": "user", "content": user_prompt}]

            # 使用带拦截机制的生成（自动检测工具调用并注入 Summary）
            orchestrator_response = await llm.generate_with_interception(
                system_prompt=combined_prompt,  # 组合的系统提示词
                messages=conversation_history,  # 完整对话历史（包含 ReAct 循环上下文）
                interceptor=interceptor
            )

            return {
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "orchestrator_response": orchestrator_response,
                    "skill_summary": skill_summary
                }
            }

        except Exception as e:
            logger.error(f"Error in orchestrator node: {e}")
            return {
                "errors": [f"Orchestrator error: {str(e)}"]
            }

    async def _intercept_instructions_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        截获并解析Worker指令节点

        从Orchestrator的回复中提取Worker指令（使用新的 SkillInterceptor）

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含解析出的Worker指令列表
        """
        logger.debug("Executing intercept_instructions node")

        try:
            # 获取Orchestrator的回复
            orchestrator_response = state.get("shared_knowledge", {}).get("orchestrator_response", "")

            # 使用新的 SkillInterceptor 解析指令
            from meta_agent.core.skill_interceptor import SkillInterceptor
            from pathlib import Path

            skills_dir = Path("meta_agent/skills")
            interceptor = SkillInterceptor(skills_dir)

            # 检测所有工具调用
            instructions = []
            search_text = orchestrator_response
            last_end_pos = 0

            while True:
                tool_call_info = interceptor.intercept_tool_call(search_text)
                if not tool_call_info:
                    break

                # 验证格式
                if not tool_call_info.is_valid:
                    logger.warning(f"Invalid tool call format: {tool_call_info.error_message}")
                    # 跳过这个无效的调用，继续搜索
                    search_text = search_text[tool_call_info.end_pos:]
                    last_end_pos += tool_call_info.end_pos
                    continue

                # 添加到指令列表
                instructions.append({
                    "worker_name": tool_call_info.tool_name,
                    "input_context": tool_call_info.parameters
                })

                # 继续检测剩余文本
                search_text = search_text[tool_call_info.end_pos:]
                last_end_pos += tool_call_info.end_pos

            logger.debug(f"Parsed {len(instructions)} worker instructions using SkillInterceptor")

            return {
                "worker_instructions": instructions,
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "worker_count": len(instructions)
                }
            }

        except Exception as e:
            logger.error(f"Error in intercept_instructions node: {e}", exc_info=True)
            return {
                "errors": [f"Instruction parsing error: {str(e)}"]
            }

    async def _execute_workers_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        执行所有Worker节点

        遍历Worker指令列表，执行每个Worker

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含所有Worker的执行结果
        """
        logger.debug("Executing execute_workers node")

        try:
            # 获取Worker指令列表
            instructions = state.get("worker_instructions", [])

            if not instructions:
                return {
                    "errors": ["No worker instructions to execute"]
                }

            # 使用SkillExecutor的execute_workers方法批量执行
            results_summary = await self.skill_agent.skill_executor.execute_workers(
                instructions,
                skill_name="content-generate"
            )

            logger.debug(f"Workers execution completed: {results_summary['successful']}/{results_summary['total_workers']} successful")

            return {
                "worker_results": results_summary["results"],
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "workers_executed": True,
                    "successful_workers": results_summary["successful"],
                    "failed_workers": results_summary["failed"]
                }
            }

        except Exception as e:
            logger.error(f"Error in execute_workers node: {e}")
            return {
                "errors": [f"Workers execution error: {str(e)}"]
            }

    async def _save_to_local_node(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        保存到本地节点

        将Worker生成的内容保存到本地文件

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含保存结果
        """
        logger.debug("Executing save_to_local node")

        try:
            import os
            from datetime import datetime

            # 检查是否需要保存
            if not state.get("save_to_local", False):
                logger.debug("Save to local not requested, skipping")
                return {
                    "shared_knowledge": {
                        **state.get("shared_knowledge", {}),
                        "save_skipped": True
                    }
                }

            # 获取Worker结果
            worker_results = state.get("worker_results", {})
            if not worker_results:
                return {
                    "errors": ["No worker results to save"]
                }

            # 生成时间戳目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("output", "social_media_content", timestamp)

            # 创建目录
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")

            # 保存各平台内容
            saved_files = []
            platform_map = {
                "@Douyin_Expert": "douyin.md",
                "@Xiaohongshu_Expert": "xiaohongshu.md",
                "@Weibo_Expert": "weibo.md"
            }

            for worker_name, result in worker_results.items():
                if result.get("success") and result.get("content"):
                    filename = platform_map.get(worker_name, f"{worker_name}.md")
                    filepath = os.path.join(output_dir, filename)

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(result["content"])

                    saved_files.append(filename)
                    logger.debug(f"Saved {filename}")

            # 生成汇总报告
            summary_content = self._generate_summary_report(state, worker_results, saved_files)
            summary_path = os.path.join(output_dir, "summary.md")

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)

            saved_files.append("summary.md")
            logger.debug("Saved summary.md")

            return {
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "saved_to_local": True,
                    "output_directory": output_dir,
                    "saved_files": saved_files
                },
                "final_result": f"Content saved to: {output_dir}"
            }

        except Exception as e:
            logger.error(f"Error in save_to_local node: {e}")
            return {
                "errors": [f"Save to local error: {str(e)}"]
            }

    def _generate_summary_report(
        self,
        state: AgentState,
        worker_results: Dict[str, Any],
        saved_files: list
    ) -> str:
        """
        生成汇总报告

        Args:
            state: 当前状态
            worker_results: Worker执行结果
            saved_files: 已保存的文件列表

        Returns:
            汇总报告内容
        """
        from datetime import datetime

        report_lines = [
            "# 社交媒体内容生成报告",
            "",
            "## 基本信息",
            f"- **主题**: {state.get('user_prompt', 'N/A')}",
            f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **平台数量**: {len(worker_results)}",
            "",
            "## 生成内容",
            ""
        ]

        # 各平台内容预览
        platform_names = {
            "@Douyin_Expert": "抖音",
            "@Xiaohongshu_Expert": "小红书",
            "@Weibo_Expert": "微博"
        }

        for worker_name, result in worker_results.items():
            platform_name = platform_names.get(worker_name, worker_name)

            if result.get("success"):
                content = result.get("content", "")
                preview = content[:200] + "..." if len(content) > 200 else content

                report_lines.extend([
                    f"### {platform_name}",
                    f"- **状态**: ✅ 生成成功",
                    f"- **字数**: {len(content)} 字",
                    f"- **预览**:",
                    "```",
                    preview,
                    "```",
                    ""
                ])
            else:
                report_lines.extend([
                    f"### {platform_name}",
                    f"- **状态**: ❌ 生成失败",
                    f"- **错误**: {result.get('error', 'Unknown error')}",
                    ""
                ])

        # 文件列表
        report_lines.extend([
            "## 文件列表",
            ""
        ])

        for filename in saved_files:
            if filename != "summary.md":
                report_lines.append(f"- [{filename}](./{filename})")

        return "\n".join(report_lines)

    def _should_execute_workers(self, state: AgentState) -> Literal["execute_workers", "end"]:
        """
        判断是否有Worker指令需要执行

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        instructions = state.get("worker_instructions", [])

        if not instructions:
            logger.debug("No worker instructions found")
            return "end"

        logger.debug(f"Found {len(instructions)} worker instructions")
        return "execute_workers"

    def _should_save_to_local(self, state: AgentState) -> Literal["save_to_local", "end"]:
        """
        判断是否需要保存到本地

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("save_to_local", False):
            logger.debug("Save to local requested")
            return "save_to_local"

        logger.debug("Save to local not requested")
        return "end"

    def _format_skill_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        格式化skill的metadata为可读文本

        Args:
            metadata: skill的元数据字典

        Returns:
            格式化后的文本
        """
        lines = []

        # 基本信息
        if "name" in metadata:
            lines.append(f"**技能名称**: {metadata['name']}")
        if "description" in metadata:
            lines.append(f"**描述**: {metadata['description']}")
        if "category" in metadata:
            lines.append(f"**类别**: {metadata['category']}")
        if "version" in metadata:
            lines.append(f"**版本**: {metadata['version']}")
        if "author" in metadata:
            lines.append(f"**作者**: {metadata['author']}")

        # 标签
        if "tags" in metadata and metadata["tags"]:
            tags_str = ", ".join(metadata["tags"])
            lines.append(f"**标签**: {tags_str}")

        return "\n".join(lines) if lines else "无元数据"

    # ===== Plan-Execute 模式节点 =====

    async def _chat_response_node(
        self,
        state: PlanExecuteState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Chat Response 节点 - 直接返回对话回复

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含 final_output
        """
        logger.debug("Executing chat_response node")

        response = state["shared_knowledge"].get("router_response", "请问有什么可以帮助您的？")

        return {
            "final_output": response,
            "mode": "chat"
        }

    async def _tool_call_node(
        self,
        state: PlanExecuteState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Tool Call 节点 - 解析用户意图并调用工具

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含工具执行结果
        """
        logger.debug("Executing tool_call node")

        try:
            from meta_agent.core.llm_manager import LLMManager
            from meta_agent.tools.tool_manager import get_tool_manager

            # 获取工具管理器
            tool_manager = get_tool_manager()

            # 获取所有可用工具的信息
            all_tools = tool_manager.get_all_tools_for_llm()

            # 构建工具信息字符串
            tools_info = []
            for tool in all_tools:
                tool_desc = f"""**工具名称**: {tool['name']}
**描述**: {tool['description']}
**参数**:"""

                # 从 parameters.properties 中提取参数信息
                params = tool.get('parameters', {})
                properties = params.get('properties', {})
                required_list = params.get('required', [])

                if properties:
                    for param_name, param_info in properties.items():
                        required = "必需" if param_name in required_list else "可选"
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', '')
                        tool_desc += f"\n  - {param_name} ({param_type}): {param_desc} [{required}]"
                else:
                    tool_desc += "\n  无参数"

                tools_info.append(tool_desc)

            tools_section = "\n\n---\n\n".join(tools_info)

            # 构建 Tool Call Prompt
            tool_call_prompt = f"""你是一个智能工具调用助手。根据用户的请求，判断需要调用哪个工具，并提取相应的参数。

## 可用工具

{tools_section}

---

## 用户请求

{state['user_prompt']}

---

## 任务

1. 分析用户请求，确定需要调用哪个工具
2. 从用户请求中提取工具所需的参数
3. 输出 JSON 格式的工具调用信息

**输出格式**（必须是 JSON）:
```json
{{
    "tool_name": "工具名称",
    "parameters": {{
        "参数名1": "参数值1",
        "参数名2": "参数值2"
    }},
    "explanation": "简短说明为什么选择这个工具和这些参数"
}}
```

**示例1 - 搜索工具**:
用户请求: "搜索一下Python异步编程的最佳实践"
```json
{{
    "tool_name": "search_web",
    "parameters": {{
        "query": "Python异步编程最佳实践",
        "max_results": 10,
        "language": "zh-CN"
    }},
    "explanation": "用户需要搜索Python异步编程相关信息"
}}
```

**示例2 - Shell命令**:
用户请求: "列出当前目录的文件"
```json
{{
    "tool_name": "execute_shell",
    "parameters": {{
        "command": "ls -la",
        "timeout": 30
    }},
    "explanation": "用户需要执行ls命令列出文件"
}}
```

请直接输出 JSON，不要添加其他说明文字。"""

            # 调用 LLM 解析工具调用
            llm = LLMManager()
            tool_call_response = await llm.generate(
                system_prompt=tool_call_prompt,
                user_message=state['user_prompt'],
                temperature=0.3
            )

            # 解析 JSON 响应
            import json
            import re

            logger.debug(f"Tool call response: {tool_call_response}")

            # 提取 JSON
            tool_call_info = None

            # 方法1：直接解析
            try:
                tool_call_info = json.loads(tool_call_response)
                logger.debug("Successfully parsed tool call response as direct JSON")
            except json.JSONDecodeError:
                pass

            # 方法2：提取 JSON 对象
            if tool_call_info is None:
                json_match = re.search(r'\{[\s\S]*\}', tool_call_response)
                if json_match:
                    try:
                        tool_call_info = json.loads(json_match.group(0))
                        logger.debug("Successfully extracted JSON object from response")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse extracted JSON: {e}")

            # 方法3：提取代码块
            if tool_call_info is None:
                code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', tool_call_response)
                if code_block_match:
                    try:
                        tool_call_info = json.loads(code_block_match.group(1))
                        logger.debug("Successfully extracted JSON from code block")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from code block: {e}")

            if tool_call_info is None:
                error_msg = f"Failed to parse tool call info. Response: {tool_call_response[:200]}"
                logger.error(error_msg)
                return {
                    "final_output": "抱歉，我无法理解您的工具调用请求。",
                    "mode": "tool_call_error"
                }

            tool_name = tool_call_info.get("tool_name")
            parameters = tool_call_info.get("parameters", {})
            explanation = tool_call_info.get("explanation", "")

            logger.debug(f"Tool call: {tool_name} with parameters: {parameters}")

            # 执行工具
            result = await tool_manager.execute_tool(tool_name, parameters)

            if result.get("success"):
                # 工具执行成功
                tool_result = result.get("result")

                # 格式化结果
                if tool_name == "search_web":
                    # 搜索结果格式化
                    formatted_result = "## 搜索结果\n\n"
                    for idx, item in enumerate(tool_result, 1):
                        formatted_result += f"### {idx}. {item.get('title', 'N/A')}\n"
                        formatted_result += f"**URL**: {item.get('url', 'N/A')}\n"
                        formatted_result += f"**摘要**: {item.get('snippet', 'N/A')}\n\n"

                elif tool_name == "execute_shell":
                    # Shell 命令结果格式化
                    formatted_result = f"## 命令执行结果\n\n"
                    formatted_result += f"**返回码**: {tool_result.get('return_code', 'N/A')}\n\n"
                    formatted_result += f"**标准输出**:\n```\n{tool_result.get('stdout', '')}\n```\n\n"
                    if tool_result.get('stderr'):
                        formatted_result += f"**标准错误**:\n```\n{tool_result.get('stderr', '')}\n```\n"
                else:
                    # 其他工具的通用格式化
                    formatted_result = f"## 工具执行结果\n\n```json\n{json.dumps(tool_result, ensure_ascii=False, indent=2)}\n```"

                return {
                    "final_output": formatted_result,
                    "mode": "tool_call",
                    "shared_knowledge": {
                        **state.get("shared_knowledge", {}),
                        "tool_call_info": tool_call_info,
                        "tool_result": tool_result
                    }
                }
            else:
                # 工具执行失败
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Tool execution failed: {error_msg}")

                return {
                    "final_output": f"工具执行失败: {error_msg}",
                    "mode": "tool_call_error"
                }

        except Exception as e:
            logger.error(f"Error in tool_call node: {e}", exc_info=True)
            return {
                "final_output": f"工具调用出错: {str(e)}",
                "mode": "tool_call_error"
            }

    async def _router_node(
        self,
        state: PlanExecuteState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Router 节点 (Layer 1) - 基于 Frontmatter 判断是否需要调用 Skill

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含路由决策
        """
        logger.debug("Executing router node (Layer 1)")

        try:
            # Layer 1: 加载所有 skills 的 frontmatter
            skill_loader = self.skill_agent.skill_loader

            # 获取所有可用的 skills（这里可以扩展为动态发现）
            available_skills = ["content-generate"]  # 未来可以从目录扫描

            # 构建 skill_name 到 file_name 的映射
            skill_name_to_file = {}  # {metadata.name: file_name}

            frontmatters = []
            for file_name in available_skills:
                metadata = skill_loader.load_skill_metadata(file_name)
                if metadata:
                    # 记录映射关系
                    skill_name_to_file[metadata.name] = file_name

                    frontmatter_text = f"""
**Skill**: {metadata.name}
**File**: {file_name}
**Description**: {metadata.description}
**Category**: {metadata.category}
**Tags**: {', '.join(metadata.tags) if metadata.tags else 'N/A'}
"""
                    frontmatters.append(frontmatter_text)

            frontmatters_section = "\n---\n".join(frontmatters)

            logger.debug(f"Loaded {len(frontmatters)} skill frontmatters")

            # 构建 Router Prompt
            from meta_agent.config.agent_prompts import get_agent_prompt
            agent_prompt = get_agent_prompt()

            router_prompt = f"""{agent_prompt}

---

## 可用技能列表（Frontmatter）

{frontmatters_section}

---

## 路由决策指令

你是一个智能路由器。根据用户输入和上述可用技能的元数据，判断应该如何处理用户请求。

**用户输入**: {state['user_prompt']}

**决策规则**:
1. 如果用户的需求明确匹配某个技能的描述和标签，**且提供了该技能所需的必要参数**，返回该技能名称
2. 如果用户需要使用工具（如搜索、执行命令），返回 "TOOL_CALL"
3. 如果用户只是问候、闲聊或提出无法用现有技能处理的问题，返回 "CHAT"
4. 如果用户的需求不明确，或**缺少必要参数**（如内容生成缺少主题），返回 "CHAT" 并询问用户

**特别注意**：
- 对于 content-generate 技能，必须确认用户提供了**内容主题**，否则返回 "CHAT" 询问
- 仅凭"文案生成"、"生成内容"等模糊表述，不足以调用技能，必须返回 "CHAT" 要求用户明确主题

**可用工具**:
- search_web: 网络搜索工具，用于搜索信息
- execute_shell: 执行Shell命令

**输出格式**（必须是 JSON）:
```json
{{
    "action": "SKILL_FILE_NAME 或 TOOL_CALL 或 CHAT",
    "reason": "简短说明为什么做出这个决策",
    "response": "如果 action 是 CHAT，这里是给用户的回复内容"
}}
```

**重要**：action 字段必须使用 **File** 字段的值（文件名），而不是 Skill 字段的值（显示名称）。

**示例1 - 需要调用技能**:
```json
{{
    "action": "content-generate",
    "reason": "用户明确要求生成社交媒体内容"
}}
```

**示例2 - 需要调用工具**:
```json
{{
    "action": "TOOL_CALL",
    "reason": "用户需要搜索信息"
}}
```

**示例3 - 直接对话**:
```json
{{
    "action": "CHAT",
    "reason": "用户只是打招呼",
    "response": "你好！我是新媒体运营助手，可以帮你生成社交媒体内容。你可以告诉我需要什么主题的内容，我会为抖音、小红书、微博、Instagram等平台生成专业的文案。"
}}
```

**示例4 - 需要更多信息**:
```json
{{
    "action": "CHAT",
    "reason": "用户需求不明确",
    "response": "我可以帮你生成社交媒体内容。请告诉我：\\n1. 内容主题是什么？\\n2. 需要哪些平台的内容？（抖音、小红书、微博、Instagram）"
}}
```

请直接输出 JSON，不要添加其他说明文字。"""

            # 调用 LLM 进行路由决策
            from meta_agent.core.llm_manager import LLMManager

            llm = LLMManager()
            router_response = await llm.generate(
                system_prompt=router_prompt,
                user_message=state['user_prompt']
            )

            # 解析 JSON 响应
            import json
            import re

            logger.debug(f"Router response: {router_response}")

            # 提取 JSON
            router_decision = None

            # 方法1：直接解析
            try:
                router_decision = json.loads(router_response)
                logger.debug("Successfully parsed router response as direct JSON")
            except json.JSONDecodeError:
                pass

            # 方法2：提取 JSON 对象
            if router_decision is None:
                json_match = re.search(r'\{[\s\S]*\}', router_response)
                if json_match:
                    try:
                        router_decision = json.loads(json_match.group(0))
                        logger.debug("Successfully extracted JSON object from response")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse extracted JSON: {e}")

            # 方法3：提取代码块
            if router_decision is None:
                code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', router_response)
                if code_block_match:
                    try:
                        router_decision = json.loads(code_block_match.group(1))
                        logger.debug("Successfully extracted JSON from code block")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from code block: {e}")

            if router_decision is None:
                error_msg = f"Failed to parse router decision. Response: {router_response[:200]}"
                logger.error(error_msg)
                return {
                    "errors": [error_msg]
                }

            action = router_decision.get("action", "CHAT")
            reason = router_decision.get("reason", "")
            response = router_decision.get("response", "")

            logger.debug(f"Router decision: action={action}, reason={reason}")

            # 保存路由决策
            return {
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "router_action": action,
                    "router_reason": reason,
                    "router_response": response,
                    "selected_skill": action if action != "CHAT" else None,
                    "skill_name_to_file": skill_name_to_file  # 保存映射关系
                }
            }

        except Exception as e:
            logger.error(f"Error in router node: {e}", exc_info=True)
            return {
                "errors": [f"Router error: {str(e)}"]
            }

    def _should_continue_after_router(
        self,
        state: PlanExecuteState
    ) -> Literal["planner", "tool_call", "chat_response", "end"]:
        """
        Router 后的条件边：决定是进入 planner、tool_call 还是 chat_response

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        action = state["shared_knowledge"].get("router_action", "CHAT")

        if action == "CHAT":
            # 直接对话，进入 chat_response 节点
            logger.debug("Router decision: CHAT, going to chat_response")
            return "chat_response"
        elif action == "TOOL_CALL":
            # 需要调用工具，进入 tool_call 节点
            logger.debug("Router decision: TOOL_CALL, going to tool_call")
            return "tool_call"
        else:
            # 需要调用 skill，进入 planner
            logger.debug(f"Router decision: {action}, going to planner")
            return "planner"

    async def _planner_node(
        self,
        state: PlanExecuteState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Planner 节点 (Layer 2) - 根据用户需求和 Skill Summary 生成执行计划

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含生成的计划
        """
        logger.debug("Executing planner node (Layer 2)")

        try:
            # Layer 2: 根据 router 的决策加载对应 skill 的完整 Summary
            skill_name = state["shared_knowledge"].get("selected_skill")
            if not skill_name:
                return {
                    "errors": ["No skill selected by router"]
                }

            skill_loader = self.skill_agent.skill_loader

            logger.debug(f"Loading skill summary (Layer 2) for: {skill_name}")
            skill_data = skill_loader.load_skill_with_metadata(skill_name)
            if not skill_data:
                return {
                    "errors": [f"Failed to load skill data: {skill_name}"]
                }

            # 提取 metadata 和 summary
            metadata = skill_data.get("metadata", {})
            skill_summary = skill_data.get("summary", "")
            logger.debug(f"Skill summary loaded, length: {len(skill_summary)}")

            # 构建 Planner Prompt
            from meta_agent.config.agent_prompts import get_agent_prompt
            agent_prompt = get_agent_prompt()

            planner_prompt = f"""{agent_prompt}

---

## 可用 Worker 信息

{skill_summary}

---

## 任务规划指令

你是一个任务规划专家。根据用户需求和上述可用的 Worker，生成详细的执行计划。

**用户需求**: {state['user_prompt']}

**重要说明**：
- 如果用户的需求明确需要生成社交媒体内容，请生成具体的执行计划（JSON数组）
- 如果用户的需求不明确、是简单问候、或无法生成具体计划，请返回空数组 [] 并用自然语言说明情况

**输出要求**:
1. 分析用户需求，确定需要哪些平台的内容
2. 为每个平台生成一个任务步骤
3. 输出格式必须是 JSON 数组，每个元素包含：
   - worker_name: Worker名称（如 @Douyin_Expert）
   - input_context: 输入参数（格式：主题=xxx，风格=xxx）

**示例输出**:
```json
[
    {{"worker_name": "@Douyin_Expert", "input_context": "主题=iPhone推广，风格=科技感"}},
    {{"worker_name": "@Xiaohongshu_Expert", "input_context": "主题=iPhone推广，风格=种草"}},
    {{"worker_name": "@Weibo_Expert", "input_context": "主题=iPhone推广，风格=观点"}},
    {{"worker_name": "@Instagram_Expert", "input_context": "主题=iPhone推广，风格=生活方式"}}
]
```

如果无法生成具体计划，请返回：
```json
[]
```
并在后面用自然语言解释原因。"""

            # 调用 LLM 生成计划
            # 注意：这里必须导入并创建 LLMManager，但要确保只在需要时创建
            # 避免在模块加载时就初始化，以防止架构不兼容问题
            from meta_agent.core.llm_manager import LLMManager

            try:
                llm = LLMManager()
                plan_response = await llm.generate(
                    system_prompt=planner_prompt,
                    user_message=state['user_prompt']
                )
            except ImportError as e:
                # 如果遇到架构不兼容错误，提供更友好的错误信息
                error_msg = f"LLM initialization failed due to architecture incompatibility: {str(e)}"
                logger.error(error_msg)
                return {
                    "errors": [error_msg]
                }

            # 解析 JSON 计划
            import json
            import re

            logger.debug(f"Planner response: {plan_response}")

            # 尝试多种方式提取 JSON
            plan_json = None

            # 方法1：尝试直接解析整个响应
            try:
                plan_json = json.loads(plan_response)
                logger.debug("Successfully parsed response as direct JSON")
            except json.JSONDecodeError:
                pass

            # 方法2：提取 JSON 数组
            if plan_json is None:
                json_match = re.search(r'\[[\s\S]*\]', plan_response)
                if json_match:
                    try:
                        plan_json = json.loads(json_match.group(0))
                        logger.debug("Successfully extracted JSON array from response")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse extracted JSON: {e}")

            # 方法3：提取 ```json 代码块
            if plan_json is None:
                code_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', plan_response)
                if code_block_match:
                    try:
                        plan_json = json.loads(code_block_match.group(1))
                        logger.debug("Successfully extracted JSON from code block")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from code block: {e}")

            if plan_json is None:
                error_msg = f"Failed to parse plan from LLM response. Response: {plan_response[:200]}"
                logger.error(error_msg)
                return {
                    "errors": [error_msg]
                }

            # 转换为 TaskStep 列表
            plan: List[TaskStep] = []
            for idx, task in enumerate(plan_json):
                step = TaskStep(
                    step_id=f"step_{idx}",
                    worker_name=task["worker_name"],
                    input_context=task["input_context"],
                    status="pending"
                )
                plan.append(step)

            logger.debug(f"Generated plan with {len(plan)} steps")

            # 如果计划为空，提取 LLM 的自然语言回复
            natural_response = None
            if len(plan) == 0:
                # 尝试提取 JSON 之后的自然语言说明
                json_end = plan_response.rfind(']')
                if json_end != -1:
                    natural_response = plan_response[json_end + 1:].strip()
                    # 移除可能的代码块标记
                    natural_response = re.sub(r'```.*?```', '', natural_response, flags=re.DOTALL).strip()

                if not natural_response:
                    natural_response = "抱歉，我无法根据当前输入生成具体的内容计划。请提供更具体的需求，比如：'帮我生成关于AI技术的社交媒体内容'。"

                logger.debug(f"Empty plan, using natural language response: {natural_response[:100]}")

            return {
                "plan": plan,
                "current_step_index": 0,
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "skill_summary": skill_summary,
                    "plan_generated": True,
                    "natural_response": natural_response  # 保存自然语言回复
                }
            }

        except Exception as e:
            logger.error(f"Error in planner node: {e}", exc_info=True)
            return {
                "errors": [f"Planner error: {str(e)}"]
            }

    async def _worker_node(
        self,
        state: PlanExecuteState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Worker 节点 - 并行执行所有计划中的任务

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含所有执行结果
        """
        logger.debug("Executing worker node (parallel mode)")

        try:
            plan = state["plan"]

            # 如果计划为空，直接跳过执行
            if len(plan) == 0:
                logger.debug("Empty plan, skipping worker execution")
                return {
                    "current_step_index": len(plan)  # 设置为计划长度，表示全部完成
                }

            # 执行 Worker
            executor = self.skill_agent.skill_executor

            # 构建所有 Worker 指令
            worker_instructions = [
                {
                    "worker_name": task["worker_name"],
                    "input_context": task["input_context"]
                }
                for task in plan
            ]

            logger.debug(f"Executing {len(worker_instructions)} workers in parallel")

            # 并行执行所有 Worker
            import asyncio
            results = await asyncio.gather(*[
                executor.execute_single_worker(instruction, skill_name="content-generate")
                for instruction in worker_instructions
            ], return_exceptions=True)

            # 处理结果并更新 plan
            updated_plan = []
            step_results = {}

            for i, (task, result) in enumerate(zip(plan, results)):
                # 处理异常情况
                if isinstance(result, Exception):
                    logger.error(f"Worker {task['worker_name']} failed with exception: {result}")
                    result = {
                        "success": False,
                        "error": str(result),
                        "worker_name": task["worker_name"]
                    }

                # 更新任务状态
                if result.get("success"):
                    updated_step = TaskStep(
                        step_id=task["step_id"],
                        worker_name=task["worker_name"],
                        input_context=task["input_context"],
                        status="completed",
                        result=result
                    )
                    logger.debug(f"✅ Worker {task['worker_name']} completed successfully")
                else:
                    updated_step = TaskStep(
                        step_id=task["step_id"],
                        worker_name=task["worker_name"],
                        input_context=task["input_context"],
                        status="failed",
                        result=result
                    )
                    logger.warning(f"❌ Worker {task['worker_name']} failed: {result.get('error', 'Unknown error')}")

                updated_plan.append(updated_step)
                step_results[task["step_id"]] = result

            logger.debug(f"All {len(worker_instructions)} workers completed")

            return {
                "plan": updated_plan,
                "step_results": step_results,
                "current_step_index": len(plan)  # 设置为计划长度，表示全部完成
            }

        except Exception as e:
            logger.error(f"Error in worker node: {e}", exc_info=True)
            return {
                "errors": [f"Worker error: {str(e)}"]
            }

    async def _replan_node(
        self,
        state: PlanExecuteState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Replan 节点 - 检查执行结果并决定下一步

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含是否需要重新规划的决策
        """
        logger.debug("Executing replan node")

        try:
            current_index = state["current_step_index"]
            plan = state["plan"]
            step_results = state.get("step_results", {})

            # 如果计划为空，使用自然语言回复
            if len(plan) == 0:
                logger.debug("Empty plan, using natural language response")
                natural_response = state["shared_knowledge"].get("natural_response", "请提供更具体的需求。")

                return {
                    "final_output": natural_response,
                    "need_replan": False
                }

            # 检查是否所有任务都完成
            if current_index >= len(plan):
                logger.debug("All steps completed, generating final output")

                # 生成最终输出
                final_output = self._generate_final_output(state)

                return {
                    "final_output": final_output,
                    "need_replan": False
                }

            # 检查上一步是否失败
            if current_index > 0:
                last_step = plan[current_index - 1]
                last_result = step_results.get(last_step["step_id"])

                if last_result and not last_result.get("success"):
                    logger.warning(f"Step {current_index - 1} failed, need replan")
                    return {
                        "need_replan": True,
                        "replan_reason": f"Step {last_step['step_id']} failed: {last_result.get('error', 'Unknown error')}"
                    }

            # 继续执行下一步
            logger.debug(f"Continuing to next step: {current_index}")
            return {
                "need_replan": False
            }

        except Exception as e:
            logger.error(f"Error in replan node: {e}", exc_info=True)
            return {
                "errors": [f"Replan error: {str(e)}"]
            }

    def _generate_final_output(self, state: PlanExecuteState) -> str:
        """
        生成最终输出

        Args:
            state: 当前状态

        Returns:
            最终输出文本
        """
        step_results = state.get("step_results", {})
        plan = state["plan"]

        output_lines = ["✅ 内容生成完成！\n", "📊 生成结果："]

        platform_names = {
            "@Douyin_Expert": "抖音短视频脚本",
            "@Xiaohongshu_Expert": "小红书种草笔记",
            "@Weibo_Expert": "微博短文本",
            "@Instagram_Expert": "Instagram帖子"
        }

        for step in plan:
            result = step_results.get(step["step_id"])
            platform_name = platform_names.get(step["worker_name"], step["worker_name"])

            if result and result.get("success"):
                content = result.get("content", "")
                word_count = len(content)
                output_lines.append(f"- {platform_name}：已生成（约{word_count}字）")
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result"
                output_lines.append(f"- {platform_name}：生成失败（{error_msg}）")

        return "\n".join(output_lines)

    async def _qa_node(
        self,
        state: PlanExecuteState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        QA 质量检查节点 - 通过 LLM 进行质量检查并生成汇总报告

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含 QA 报告
        """
        logger.debug("Executing QA node")

        try:
            plan = state["plan"]
            step_results = state.get("step_results", {})
            user_query = state.get("user_query", "")

            # 构建 QA 提示词
            qa_prompt = self._build_qa_prompt(user_query, plan, step_results)

            # 调用 LLM 进行质量检查
            from meta_agent.core.llm_manager import LLMManager

            llm = LLMManager()
            logger.debug("Calling LLM for QA check")
            qa_report = await llm.generate(
                system_prompt="你是一位专业的内容质量审核专家。",
                user_message=qa_prompt,
                temperature=0.3  # 使用较低温度以获得更稳定的输出
            )

            logger.debug("QA check completed")

            # 不修改 final_output，将 QA 报告单独存储
            # 这样可以在输出时灵活控制顺序
            return {
                "qa_report": qa_report
            }

        except Exception as e:
            logger.error(f"Error in QA node: {e}", exc_info=True)
            # QA 失败不影响主流程，返回原始输出
            return {
                "qa_report": f"QA 检查失败: {str(e)}"
            }

    def _build_qa_prompt(
        self,
        user_query: str,
        plan: list,
        step_results: Dict[str, Any]
    ) -> str:
        """
        构建 QA 提示词

        Args:
            user_query: 用户原始查询
            plan: 执行计划
            step_results: 步骤执行结果

        Returns:
            QA 提示词
        """
        # 收集所有生成的内容
        platform_contents = {}
        platform_names = {
            "@Douyin_Expert": "抖音",
            "@Xiaohongshu_Expert": "小红书",
            "@Weibo_Expert": "微博",
            "@Instagram_Expert": "Instagram"
        }

        for step in plan:
            result = step_results.get(step["step_id"])
            if result and result.get("success"):
                platform_name = platform_names.get(step["worker_name"], step["worker_name"])
                content = result.get("content", "")
                platform_contents[platform_name] = content

        # 构建内容预览
        content_preview = []
        for platform, content in platform_contents.items():
            word_count = len(content)
            preview = content[:200] + "..." if len(content) > 200 else content
            content_preview.append(f"**{platform}** ({word_count}字):\n{preview}\n")

        content_preview_text = "\n".join(content_preview)

        prompt = f"""你是一位专业的内容质量审核专家，请对以下社交媒体内容生成结果进行质量检查和汇总。

## 用户需求
{user_query}

## 生成的内容预览
{content_preview_text}

## 请完成以下任务

### 1. 质量评估
请从以下维度评估内容质量（每项1-5分）：
- **内容相关性**: 是否符合用户需求
- **平台适配性**: 是否符合各平台特点
- **创意性**: 内容是否有吸引力
- **完整性**: 内容是否完整、结构清晰
- **语言质量**: 文字表达是否流畅、无错别字

### 2. 优缺点分析
- **优点**: 列出内容的亮点（2-3点）
- **待改进**: 指出可以优化的地方（如有）

### 3. 整体评价
用1-2句话总结整体质量。

### 4. 建议
如果有需要改进的地方，给出具体建议。

请以清晰、专业的方式组织你的回答，使用 Markdown 格式。"""

        return prompt

    def _should_save_after_qa(
        self,
        state: PlanExecuteState
    ) -> Literal["save_to_local", "end"]:
        """
        QA 后的条件边：决定是否保存到本地

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        # 检查是否需要保存到本地
        if state.get("save_to_local", False):
            logger.debug("Need to save to local, going to save_to_local")
            return "save_to_local"
        else:
            logger.debug("No need to save, going to end")
            return "end"

    def _should_continue_after_replan(
        self,
        state: PlanExecuteState
    ) -> Literal["worker", "planner", "qa", "end"]:
        """
        Replan 后的条件边：决定下一步

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        # 如果有最终输出，说明所有任务完成，进入 QA 节点
        if state.get("final_output"):
            logger.debug("All tasks completed, going to QA node")
            return "qa"

        # 如果需要重新规划，回到 planner
        if state.get("need_replan"):
            logger.debug("Need replan, going back to planner")
            return "planner"

        # 否则继续执行下一个 worker
        logger.debug("Continuing to next worker")
        return "worker"



    async def _save_to_local_plan_execute_node(
        self,
        state: PlanExecuteState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        保存到本地节点（Plan-Execute 模式）

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新
        """
        logger.debug("Executing save_to_local node (Plan-Execute)")

        try:
            import os
            from datetime import datetime

            step_results = state.get("step_results", {})
            if not step_results:
                return {
                    "errors": ["No results to save"]
                }

            # 生成时间戳目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("output", "social_media_content", timestamp)

            # 创建目录
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")

            # 保存各平台内容
            saved_files = []
            platform_map = {
                "@Douyin_Expert": "douyin.md",
                "@Xiaohongshu_Expert": "xiaohongshu.md",
                "@Weibo_Expert": "weibo.md",
                "@Instagram_Expert": "instagram.md"
            }

            for step in state["plan"]:
                result = step_results.get(step["step_id"])
                if result and result.get("success") and result.get("content"):
                    filename = platform_map.get(step["worker_name"], f"{step['worker_name']}.md")
                    filepath = os.path.join(output_dir, filename)

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(result["content"])

                    saved_files.append(filename)
                    logger.debug(f"Saved {filename}")

            # 生成汇总报告
            summary_content = self._generate_summary_report_plan_execute(state, saved_files)
            summary_path = os.path.join(output_dir, "summary.md")

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)

            saved_files.append("summary.md")
            logger.debug("Saved summary.md")

            return {
                "output_directory": output_dir,
                "shared_knowledge": {
                    **state.get("shared_knowledge", {}),
                    "saved_files": saved_files
                }
            }

        except Exception as e:
            logger.error(f"Error in save_to_local node: {e}", exc_info=True)
            return {
                "errors": [f"Save to local error: {str(e)}"]
            }

    def _generate_summary_report_plan_execute(
        self,
        state: PlanExecuteState,
        saved_files: List[str]
    ) -> str:
        """
        生成汇总报告（Plan-Execute 模式）

        Args:
            state: 当前状态
            saved_files: 已保存的文件列表

        Returns:
            汇总报告内容
        """
        from datetime import datetime

        report_lines = [
            "# 社交媒体内容生成报告",
            "",
            "## 基本信息",
            f"- **主题**: {state.get('user_prompt', 'N/A')}",
            f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **任务数量**: {len(state['plan'])}",
            "",
            "## 执行计划",
            ""
        ]

        # 显示执行计划
        for idx, step in enumerate(state["plan"], 1):
            status_emoji = "✅" if step["status"] == "completed" else "❌" if step["status"] == "failed" else "⏳"
            report_lines.append(f"{idx}. {status_emoji} {step['worker_name']} - {step['status']}")

        report_lines.extend(["", "## 生成内容", ""])

        # 各平台内容预览
        platform_names = {
            "@Douyin_Expert": "抖音",
            "@Xiaohongshu_Expert": "小红书",
            "@Weibo_Expert": "微博",
            "@Instagram_Expert": "Instagram"
        }

        step_results = state.get("step_results", {})
        for step in state["plan"]:
            platform_name = platform_names.get(step["worker_name"], step["worker_name"])
            result = step_results.get(step["step_id"])

            if result and result.get("success"):
                content = result.get("content", "")
                preview = content[:200] + "..." if len(content) > 200 else content

                report_lines.extend([
                    f"### {platform_name}",
                    f"- **状态**: ✅ 生成成功",
                    f"- **字数**: {len(content)} 字",
                    f"- **预览**:",
                    "```",
                    preview,
                    "```",
                    ""
                ])
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result"
                report_lines.extend([
                    f"### {platform_name}",
                    f"- **状态**: ❌ 生成失败",
                    f"- **错误**: {error_msg}",
                    ""
                ])

        # 文件列表
        report_lines.extend(["## 文件列表", ""])
        for filename in saved_files:
            if filename != "summary.md":
                report_lines.append(f"- [{filename}](./{filename})")

        return "\n".join(report_lines)
